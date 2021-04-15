from nltk.stem.snowball import SnowballStemmer
from os.path import dirname, realpath, isfile
import json
from time import time
import re
from nltk import word_tokenize
from math import log10, log, sqrt

N_DOCUMENT = 7945
N_RETURN = 20  # only return the top N_RETURN result for the search

stemmer = SnowballStemmer("english")
english_words = re.compile('[A-Za-z]+')

document_len = dirname(realpath(__file__)) + '/index/doclen.txt'
index_file = dirname(realpath(__file__)) + '/index/index.txt'
data_dir = dirname(realpath(__file__)) + '/HillaryEmails'

def get_doc_from_id(doc_id):
    return data_dir + '/%s.txt' % doc_id

# removed numbers, symbols, non-english chars
def tokenize(string):
    tokens = [w.lower() for w in word_tokenize(string) if
              english_words.fullmatch(w)]
    return tokens

# pre-process query to a list of searchable terms
def pre_process(string):
    term_list = [stemmer.stem(t) for t in tokenize(string)]
    return term_list

def load_document_length():
    ret = {}
    if isfile(document_len):
        with open(document_len, 'r') as f:
            ret = json.loads(f.read())
            print('load_document_length [%s] docs' %len(ret))
        return ret
    for i in range(1, N_DOCUMENT + 1):
        with open(get_doc_from_id(i), 'r') as f:
            ret[i] = len(tokenize(f.read()))
    with open(document_len, 'w') as f:
        f.write(json.dumps(ret))
    print('load_document_length [%s] docs' %len(ret))
    return ret

def load_posting():
    posting = {}
    with open(index_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if l:
                for k, v in json.loads(l).items():
                    posting[k] = v
    print('load_posting [%s] pairs' %len(posting))
    return posting

def operation_or(l1, l2):
    i, j = 0, 0
    result = []
    if not l1:
        return l2
    if not l2:
        return l1
    max1 = len(l1) - 1
    max2 = len(l2) - 1
    while True:
        if l1[i][0] > l2[j][0]:
            result.append(l2[j])
            if j != max2:
                j += 1
            else:
                if result and l1[i][0] == result[-1][0]:
                    result.pop()
                result.extend(l1[i:])
                break
        elif l2[j][0] > l1[i][0]:
            result.append(l1[i])
            if i != max1:
                i += 1
            else:
                if result and l2[j][0] == result[-1][0]:
                    result.pop()
                result.extend(l2[j:])
                break
        if l1[i][0] == l2[j][0]:
            result.append(l1[i])
            if i != max1:
                i += 1
            else:
                if result and l2[j][0] == result[-1][0]:
                    result.pop()
                result.extend(l2[j:])
                break
            if j != max2:
                j += 1
            else:
                if result and l1[i][0] == result[-1][0]:
                    result.pop()
                result.extend(l1[i:])
                break
    return result

# binary search: check if the given document id can be found in the term-posting list
# return term-freq if corresponding document id is found, otherwise return 0
def binsearch_tf(plst, docid):
    if not plst:
        return 0
    low = 0
    high = len(plst) - 1
    while low <= high:
        mid = (low + high) // 2
        if plst[mid][0] < docid:
            low = mid + 1
        elif plst[mid][0] > docid:
            high = mid - 1
        else:
            return plst[mid][1]
    return 0

def query_vector(terms, idf):
    # (1 + log(tf)) x log(N/df)
    # build a simple freq dist dict for terms
    fdd = {}
    for t in terms:
        fdd[t] = fdd.get(t, 0) + 1
    return [(1 + log10(fdd[t])) * idf.get(t, 0) for t in terms]

def document_vector(terms, idf, docid, pos):
    vec = []
    for t in terms:
        tf = binsearch_tf(pos.get(t, []), docid)
        if not tf:
            vec.append(0)
        else:
            vec.append((1 + log10(tf)) * idf.get(t, 0))
    return vec

# cosine similarity for query vector and document vector
def cos_sim(qv, dv):
    if len(qv) != len(dv):
        raise ValueError('Err length qv [%s] dv [%s]' %(len(qv), len(dv)))
    if not dv:
        return 0

    # check if document vector values are all 0
    chk = True
    for v in dv:
        if v != 0:
            chk = False
            break
    if chk:
        return 0

    mod_qv, mod_dv, sum_p = 0, 0 , 0
    for n1, n2 in zip(qv, dv):
        sum_p += n1 * n2
        mod_qv += n1 * n1
        mod_dv += n2 * n2

    return sum_p / (sqrt(mod_qv) * sqrt(mod_dv))

def search(query, pos, doc):
    terms = pre_process(query)
    print('search %s' %terms)
    result = []
    idf = {}
    ranking = {}
    for t in terms:
        poslist = pos.get(t, [])
        idf[t] = 0
        if poslist:
            idf[t] = log10(N_DOCUMENT / len(poslist))
        result = operation_or(result, poslist)

    qvec = query_vector(terms, idf)
    for r in result:
        docid = r[0]
        dvec = document_vector(terms, idf, docid, pos)
        ranking[docid] = cos_sim(qvec, dvec) / log(max(16, doc.get('%s' %docid)), 16)

    print('ranking list length %s' %len(ranking))
    return sorted(ranking.items(), key=lambda tup: tup[1], reverse=True)[:N_RETURN]

if __name__ == '__main__':
    posting = load_posting()
    doclen = load_document_length()

    text = "Please convey to the Secretary my most profound thanks for her contribution to the 'It Gets Better' video series."

    start = time()
    print(search(text, posting, doclen))   # returns [(6838, 0.5356845360048234), (4054, 0.3555446148043571), (6868, 0.3442583293694912) ....]
    print('time spent [%s]' %(time() - start))