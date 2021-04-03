from nltk.stem.snowball import SnowballStemmer
from os.path import dirname, realpath
from collections import OrderedDict
import json
from memutil import total_size
from time import time

index_file = dirname(realpath(__file__)) + '/index.txt'
index_file_test = dirname(realpath(__file__)) + '/index_test.txt'

index_uncompressed = OrderedDict()
index_compressed = OrderedDict()

stemmer = SnowballStemmer("english")

def load_index_uncompressed(fname):
    if index_uncompressed:
        return
    start = time()
    with open(fname) as f:
        lines = f.readlines()
        for l in lines:
            j = json.loads(l)
            for k, v in j.items():
                posting = [p[0] for p in v]
                index_uncompressed[k] = posting
    end = time()
    print('load_index_uncompressed: [%s] terms-posting pairs [%s] bytes in [%s] sec'
          %(len(index_uncompressed), total_size(index_uncompressed), end - start))

def load_index_compressed(fname):
    if index_compressed:
        return
    start = time()
    with open(fname) as f:
        lines = f.readlines()
        terms = b''
        index = 0
        term_hash = {}
        hash_key = -1
        for l in lines:
            j = json.loads(l)
            for k, v in j.items():
                posting = b'.'.join(('%x' %p[0]).encode() for p in v)
                term = k.encode()
                if term[0] != hash_key:
                    hash_key = term[0]
                    term_hash[hash_key] = index
                terms += b'.%s' %term
                index_compressed[('%x' %index).encode()] = posting.strip(b'.')
                index += (len(k) + 1)
        index_compressed['terms'] = terms.strip(b'.')
        index_compressed['term_hash'] = term_hash
    end = time()
    print('load_index_compressed: [%s] terms-posting pairs [%s] bytes in [%s] sec'
          %(len(index_compressed) - 2 if len(index_compressed) else 0, total_size(index_compressed), end - start))

def stem(token):
    return stemmer.stem(token.lower())

def op_not(l1, l2):
    i, j = 0, 0
    result = []
    if not l2:
        return l1
    if not l1:
        return result
    max1 = len(l1) - 1
    max2 = len(l2) - 1
    while True:
        if l1[i] > l2[j]:
            if j != max2:
                j += 1
            else:
                result.extend(l1[i:])
                break
        elif l2[j] > l1[i]:
            if i != max1:
                result.append(l1[i])
                i += 1
            else:
                result.extend(l1[i:])
                break
        if l1[i] == l2[j]:
            if i != max1:
                i += 1
            else:
                break
            if j != max2:
                j += 1
            else:
                result.extend(l1[i:])
                break
    return result

def op_or(l1, l2):
    i, j = 0, 0
    result = []
    if not l1:
        return l2
    if not l2:
        return l1
    max1 = len(l1) - 1
    max2 = len(l2) - 1
    while True:
        if l1[i] > l2[j]:
            result.append(l2[j])
            if j != max2:
                j += 1
            else:
                if result and l1[i] == result[-1]:
                    result.pop()
                result.extend(l1[i:])
                break
        elif l2[j] > l1[i]:
            result.append(l1[i])
            if i != max1:
                i += 1
            else:
                if result and l2[j] == result[-1]:
                    result.pop()
                result.extend(l2[j:])
                break
        if l1[i] == l2[j]:
            result.append(l1[i])
            if i != max1:
                i += 1
            else:
                if result and l2[j] == result[-1]:
                    result.pop()
                result.extend(l2[j:])
                break
            if j != max2:
                j += 1
            else:
                if result and l1[i] == result[-1]:
                    result.pop()
                result.extend(l1[i:])
                break
    return result

def op_and(l1, l2):
    i, j = 0, 0
    result = []
    if (not l1) or (not l2):
        return result
    max1 = len(l1) - 1
    max2 = len(l2) - 1
    while True:
        if l1[i] > l2[j]:
            if j != max2:
                j += 1
            else:
                break
        elif l2[j] > l1[i]:
            if i != max1:
                i += 1
            else:
                break
        if l1[i] == l2[j]:
            result.append(l1[i])
            if i != max1:
                i += 1
            else:
                break
            if j != max2:
                j += 1
            else:
                break
    return result

def operation(lst1, lst2, op):
    if op == 'AND':
        return op_and(lst1, lst2)
    elif op == 'OR':
        return op_or(lst1, lst2)
    elif op == 'NOT':
        return op_not(lst1, lst2)
    raise ValueError('invalid op [%s]' %op)

def get_posting(term, compress=False):
    if not compress:
        if not index_uncompressed:
            load_index_uncompressed(index_file)
        return index_uncompressed.get(term, [])

    if not index_compressed:
        load_index_compressed(index_file)

    # get the sub byte-string that may contain the searched term
    term = term.encode()
    term_hash = index_compressed['term_hash']
    idx_lo = term[0]
    idx_hi = idx_lo
    idx_max = ord(b'z') + 1
    if not idx_lo in term_hash:
        return []
    while idx_hi < idx_max:
        idx_hi += 1
        if idx_hi in term_hash:
            break
    if idx_hi < idx_max:
        substr = index_compressed['terms'][term_hash[idx_lo]:term_hash[idx_hi]]
    else:
        substr = index_compressed['terms'][term_hash[idx_lo]:]

    # find key for term-posting
    ss = substr.strip(b'.').split(b'.')
    k = term_hash[idx_lo]
    found = False
    for s in ss:
        if s == term:
            found = True
            break
        k += (len(s) + 1)
    if not found:
        return []

    # retrieve and decompress posting list
    posting = [int(i, 16) for i in index_compressed[('%x' %k).encode()].split(b'.')]
    return posting

def process_query(lst, op, term, compress=False):
    l1 = lst
    l2 = get_posting(term, compress)
    return operation(l1, l2, op)

def search(query, compress=False):
    start = time()
    tokens = query.strip().split(' ')
    if len(tokens) == 2:
        raise ValueError('invalid query')
    i = 0
    term = stem(tokens[i])
    result = process_query([], 'OR', term, compress)
    i += 1
    while i < len(tokens) - 1:
        op = tokens[i]
        i += 1
        term = stem(tokens[i])
        result = process_query(result, op, term, compress)
        i += 1
    print('search [%s] took [%s] sec' %(query, time() - start))
    return result

def test_compression():
    """
    apple [30, 90]
    apple OR blackberry [30, 80, 90, 110]
    apple OR blackberry NOT durian [30, 80, 110]
    apple OR blackberry NOT durian AND banana [30]
    """
    print('\nuncompressed index structure:')
    load_index_uncompressed(index_file_test)
    print(index_uncompressed)
    print(search('blackberry', compress=False))
    print(search('apple OR blackberry NOT durian AND banana', compress=False))

    print('\ncompressed index structure:')
    load_index_compressed(index_file_test)
    print(index_compressed)
    print(search('blackberry', compress=True))
    print(search('apple OR blackberry NOT durian AND banana', compress=True))

if __name__ == '__main__':
    #test_compression()

    print('\n')
    load_index_uncompressed(index_file)
    print('\n')
    print(search('Hillary AND Clinton AND life AND trip', compress=False),'\n')
    print(search('Hillary NOT Clinton OR life AND trip', compress=False), '\n')
    print(search('Hillary AND zzq', compress=False), '\n')

    print('\n')
    load_index_compressed(index_file)
    print('\n')
    print(search('Hillary AND Clinton AND life AND trip', compress=True),'\n')
    print(search('Hillary NOT Clinton OR life AND trip', compress=True), '\n')
    print(search('Hillary AND zzq', compress=True), '\n')

