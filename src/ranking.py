"""Ranking of documents given a query, using indexes created in build_index.py

Attributes:
    AVGDL (float): Average document length
    NUM_DOCS (int): Number of documents in collection
    NUM_TOKENS (int): Number of terms in collection
"""

from multiprocessing import Process, Manager, Pool, cpu_count
import subprocess
import copy
import heapq

from math import log
from tqdm import tqdm

import _pickle as pickle
import text_manipulation as tm
import filter_indexes as fi

# Hardcoded for convenience
NUM_DOCS = 524000
AVGDL = 426.2197061068702
NUM_TOKENS = 223339126


def retrieve_top_k(k, query, ranking_function, inv_idx, doc_freq_idx, doc_len_idx, term_freq_idx, results, key, argdict={}):
    """Retrieves ordered top k documents for a query using document-at-a-time strategy
    
    Args:
        k (int): Number of documents to retrieve
        query (str): Topic to retrieve documents for
        ranking_function (function): Function to be passed for ranking documents
        inv_idx (dict): Inverted index of corpus
        doc_freq_idx (dict): term - doc_freq
        doc_len_idx (dict): doc_id - doc_len
        term_freq_idx (TYPE): Description
        results (TYPE): Description
        key (TYPE): Description
        argdict (dict, optional): Parameters for the ranking function
    
    """

    doc_ids = sorted(doc_len_idx.keys())
    query_terms = tm.process_text(query)
    top_ranked = []

    term_postings = [sorted(inv_idx[t]) for t in query_terms]
    posting_iters = [iter(t_p) for t_p in term_postings]

    next_postings = [next(p) for p in posting_iters]
    total_term_freqs = [term_freq_idx[term] for term in query_terms]
    doc_freqs = [doc_freq_idx[term] for term in query_terms]
    for doc_id in doc_ids:
        term_freqs = []
        doc_len = doc_len_idx[doc_id]

        for i, next_p in enumerate(next_postings):
            term_freq = 0
            if doc_id == next_p[0]:
                term_freq = next_p[1]
                next_postings[i] = next(posting_iters[i], (None, None))

            term_freqs.append(term_freq)

        doc_score = ranking_function(term_freqs, doc_freqs, doc_len, total_term_freqs, argdict=argdict)

        if len(top_ranked) < k:
            heapq.heappush(top_ranked, (doc_score, doc_id))
        else:
            heapq.heappushpop(top_ranked, (doc_score, doc_id))

    results[key] = list(reversed(sorted([(score, doc_id) for score, doc_id in top_ranked])))


def query_likelihood(term_freqs, doc_freqs, doc_len, total_term_freqs, argdict={}):
    """Query likelihood retrieval score for one query and one document
    
    Args:
        term_freqs (list of int): List of frequencies of terms in query
        doc_freqs (list of int): List of binary query term occurences in documents in corpus
        doc_len (int): Length of the document
        total_term_freqs (list of int): List of total frequences of term in query
        argdict (dict, optional): Parameters for the ranking function
    
    Returns:
        float: Score of single document for query
    
    """
    smoothing = argdict.get('smoothing', 'dir')
    lambda_coeff = argdict.get('lambda', 0.8)
    mu = argdict.get('mu', 2000)

    doc_score = 0
    for term_freq, doc_freq, total_term_freq in zip(term_freqs, doc_freqs, total_term_freqs):
        p_w_c = (total_term_freq+1) / NUM_TOKENS
        if smoothing == 'jm':
            doc_score += log(1 + ((1 - lambda_coeff)*(term_freq)/((doc_len+1)) / (lambda_coeff*p_w_c)))
        elif smoothing == 'dir':
            doc_score += log(1 + (term_freq/(mu * p_w_c))) + log(mu/(mu + doc_len))
        else:
            # Naïve QL, with zero-frequency problem
            if term_freq == 0:
                doc_score -= 999
            else:
                doc_score += log(doc_len/term_freq)

    return doc_score


def okapi_bm25(term_freqs, doc_freqs, doc_len, total_term_freqs, argdict={}):
    """Vector space mode score for one query (multiple terms) and one document
    
    Args:
        term_freqs (list of int): Term occurence count in the document
        doc_freqs (list of int): Binary term occurence count in documents in corpus
        doc_len (int): Length of the document
        total_term_freqs (list of int): List of total frequences of term in query
        argdict (dict, optional): Parameters for the ranking function
    
    Returns:
        float: Score of single document for query
    
    """
    k1 = argdict.get("k1", 1.2)
    b  = argdict.get("b", 0.75)

    doc_score = 0
    for term_freq, doc_freq in zip(term_freqs, doc_freqs):
        idf = log((NUM_DOCS - doc_freq + 0.5) / (doc_freq + 0.5))
        top = term_freq * (k1 + 1)
        bot = term_freq + k1 * (1 - b + b * (doc_len/ AVGDL))

        term_score = idf * (top / bot)

        doc_score += term_score

    return doc_score


def chunks(l, n):
    """Used in multiprocessing
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def ql(tfs, dfs, dl, total_term_freqs, argdict={}):
    """Hacky way of not using lambda function
    
    Args:
        tfs (list of int): Term frequencies of query terms
        dfs (list of int): Document frequencies of query terms
        dl (int): Document length
        total_term_freqs (list of int): List of total frequences of term in query
        argdict (dict, optional): Parameters for the ranking function
    
    Returns:
        float: Result of scoring function using the parameters
    """
    return query_likelihood(tfs, dfs, dl, total_term_freqs, argdict)


def bm25(tfs, dfs, dl, total_term_freqs, argdict={}):
    """Hacky way of not using lambda function
    
    Args:
        tfs (list of int): Term frequencies of query terms
        dfs (list of int): Document frequencies of query terms
        dl (int): Document length
        total_term_freqs (list of int): List of total frequences of term in query
        argdict (dict, optional): Parameters for the ranking function
    
    Returns:
        float: Result of scoring function using the parameters
    """
    return okapi_bm25(tfs, dfs, dl, total_term_freqs, argdict)


def rank_evaluate(argdict={}):
    """Main function, used when this file is called
    
    Args:
        argdict (dict, optional): Parameters for the ranking function
    """

    print('Loading indices...')
    with open("../data/FILTERED_INVERTED_INDEX_NOSTOP.pkl", "rb") as file:
        inverted_index = pickle.load(file)
    with open("../data/DOC_LEN_INDEX_NOSTOP.pkl", "rb") as file:
        doc_len_index = pickle.load(file)
    with open("../data/DOC_FREQ_INDEX_NOSTOP.pkl", "rb") as file:
        doc_freq_index = pickle.load(file)
    with open("../data/TOPIC_DICT_NOSTOP.pkl", "rb") as file:
        topic_dict = pickle.load(file)
    with open("../data/TERM_FREQ_NOSTOP.pkl", "rb") as file:
        term_freq_idx = pickle.load(file)

    print('Finished loading indices...')

    # k = argdict.get('k', 1000)
    # argdict['fun'] = bm25
    # function = argdict.get('fun', bm25)

    with Manager() as manager:
        results = manager.dict({})

        processes = []
        for key in sorted(list(topic_dict.keys())):
            topic = topic_dict[key]
            terms = tm.process_text(topic)

            filt_inv_index = fi.filter_dict(inverted_index, terms)
            filt_doc_freq_index = fi.filter_dict(doc_freq_index, terms)
            p = Process(target=retrieve_top_k, args=[k, topic, function, filt_inv_index, filt_doc_freq_index, doc_len_index, term_freq_idx, results, key, argdict])
            processes.append(p)
        print('Build processes')
        for i in tqdm(chunks(processes,cpu_count())):
            for j in i:
                j.start()
            for j in i:
                j.join()  
        results = dict(results)
    

    trec_eval_out = []
    for num in sorted(results):
        start_str = num + ' Q0 '
        end_str = ' STANDARD'

        for i, result in enumerate(results[num]):
            trec_eval_line = start_str + ' '.join([result[1], str(i), str(result[0])]) + end_str
            trec_eval_out.append(trec_eval_line)

    out_file_name = '../outputs/output'
    out_file_name += '_' + argdict.get('fun').__name__
    for key in ['smoothing', 'lambda', 'mu', 'b', 'k1']:
        if argdict.get(key, None):
            out_file_name += '_' + key + '_' +  str(argdict.get(key))
    out_file_name += '.txt'
    with open(out_file_name, 'w+') as file:
        for line in trec_eval_out:
            print(line, file=file)
    print('Ranking complete. Result stored in ' + out_file_name)


def main():
    """Executes a simple retrieval when this file is called
    """
    rank_evaluate()


if __name__ == '__main__':
    main()
