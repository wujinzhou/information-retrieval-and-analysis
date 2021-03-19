from os.path import dirname, realpath
from bisect import insort
from collections import OrderedDict
from nltk import word_tokenize, FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from time import time
import json
import re
#from memory_profiler import profile

base_dir = dirname(realpath(__file__))
data_dir = base_dir + '/HillaryEmails'
index_dir = base_dir + '/index'

n_documents = 7945
max_blk_file = 1000  # number of doc per block
max_merge_bytes = 100000 # number of chars per flush

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))
english_words = re.compile('[A-Za-z]+')

def get_doc_from_id(doc_id):
    return data_dir + '/%s.txt' % doc_id

def get_blk_from_id(blk_id):
    return index_dir +'/blk%s.txt' % blk_id

def write_file(data, name, mode):
    file_path = index_dir + '/%s' % name
    with open(file_path, mode) as outfile:
        print('writing [%s] %s ...' %(mode, name))
        outfile.write(data)

def pre_process(doc_id):
    file_name = get_doc_from_id(doc_id)
    with open(file_name, 'r') as file:
        file_content = file.read()
        tokens = [w.lower() for w in word_tokenize(file_content) if english_words.fullmatch(w)]  # removed numbers, symbols, non-english chars
        term_list = [stemmer.stem(t) for t in tokens]
        fd = FreqDist(term_list)
        return fd, len(term_list)

class Term:
    def __init__(self, term, doc_id, tf):  #init with stemmed term!
        self.term = term
        self.posting = [(doc_id, tf)]

    def add_posting_sort(self, doc_id, tf):
        insort(self.posting, (doc_id, tf))

    def add_posting_append(self, doc_id, tf): # if document sequence is already in order, then no need to sort again
        self.posting.append((doc_id, tf))

    def get_min_id(self):
        return self.posting[0][0] # return the minimum doc_id

class Block:
    def __init__(self, doc_lst, blk_id):
        self.term_posting = {}
        self.insort_keys = []  # the key in document_posting in sorted
        self.document_list = doc_lst # the document range
        self.block_id = blk_id

    def save_index(self):
        for doc_id in self.document_list:
            fd, sz = pre_process(doc_id)
            for term, tf in fd.items():
                if term in self.term_posting:
                    self.term_posting[term].add_posting_append(doc_id, tf)
                else:
                    self.term_posting[term] = Term(term, doc_id, tf)
                    insort(self.insort_keys, term)
        outstr = ''
        for k in self.insort_keys:
            outstr += '%s\n' %json.dumps({k: self.term_posting[k].posting})
        write_file(outstr, 'blk%s.txt' % self.block_id, 'w')

class BlockMerger:
    def load_next_term(self, fhid):
        fh = self.file_handlers.get(fhid, None)
        if not fh:
            print('Error: file-handler-not-found id %s' % fhid)
            raise
        line = fh.readline()
        if line:
           self.merge_stack[fhid] = json.loads(line)
        else:
            self.merge_stack.pop(fhid, None)  # file handler reaches eof
            fh.close()

    def __init__(self, block_file_ids):
        self.file_handlers = {}
        self.merge_stack = OrderedDict()  # store <handler_id : term_posting (json obj)> pairs, order matters
        file_handler_id = 0
        for bfid in block_file_ids:
            fhandler = open(get_blk_from_id(bfid), 'r')
            self.file_handlers[file_handler_id] =  fhandler
            file_handler_id += 1

        # init merge list:
        for fhid in self.file_handlers.keys():
            self.load_next_term(fhid)

    def save_index(self):  # save the merged index from blocks to disk
        curr_term = chr(127)  # the last char in ascii table which greater than any indexable term in blocks
        curr_fhids = []
        curr_posting = []
        outstr = ''
        while self.merge_stack:
            for fhid, obj in self.merge_stack.items():
                for term, posting in obj.items():
                    if term < curr_term:
                        curr_term = term
                        curr_posting = [p for p in posting]
                        curr_fhids = [fhid]
                    elif term == curr_term:
                        curr_posting.extend(posting)  # as merge_stack is an od, the later fhid always maps to a larger posting document id
                        curr_fhids.append(fhid)

            print('merge \t[%s] \t[%s]' % (curr_term, len(curr_posting)))

            if outstr:
                outstr += '\n%s' %json.dumps({curr_term: curr_posting})
            else:
                outstr += json.dumps({curr_term: curr_posting})

            for fhid in curr_fhids:
                self.load_next_term(fhid)
            curr_term = chr(127)
            curr_fhids = []
            curr_posting = []

            if len(outstr) > max_merge_bytes:
                write_file(outstr, 'index.txt', 'a')
                outstr = ''

        if outstr:
            write_file(outstr, 'index.txt', 'a')

@profile
def split_blocks():
    max_id = 0
    for i in range(1, n_documents+1, max_blk_file):
        max_id += 1
        lst = list(range(i, min(i+max_blk_file, n_documents+1)))
        block = Block(lst, max_id)
        block.save_index()
    return max_id

@profile
def merge_blocks(max_id):
    bm = BlockMerger(list(range(1, max_id + 1)))
    bm.save_index()

if __name__ == '__main__':
    start = time()
    max_block_id = split_blocks()
    time_split_block = time() - start

    merge_blocks(max_block_id)
    time_merge_block = time() - start - time_split_block

    print('summary:\n\tmax_blk_file:\t', max_blk_file, '\n\tnum_of_blocks:\t', max_block_id,
          '\n\ttime_split_block:\t', time_split_block, '\n\tmax_merge_bytes:\t',
          max_merge_bytes,'\n\ttime_merge_index:\t', time_merge_block)

