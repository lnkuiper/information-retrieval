"""Filter all the indexes based on their occurence in the search term
   This is to create a smaller index to save memory, especially when multiprocessing
"""
import gc
import heapq
import _pickle as pickle
import sys
import re 

from math import log
from tqdm import tqdm

import text_manipulation as tm


def filter_dict(dictionary, terms):
    """Filters the dictionary on given terms
    
    Args:
        dictionary (dict): The full dictionary to filter
        terms (list of strings): List of terms to remain in the dictionary
    
    Returns:
        dictionary: Returns the dictionary of which the itemvalues were present in the term list
    """
    filtered_dict = {}
    for item in dictionary.items():
        if item[0] in terms:
            filtered_dict[item[0]] = item[1]
    return filtered_dict

def split_stop_stem_topics(topics):
    """Processes topics into seperate terms
    
    Args:
        topics (dict): Dictionary of topics
    
    Returns:
        list of strings: List containing the seperate terms in the topic dicitonary
    """
    proc_topics = [tm.process_text(topic) for topic in topics.values()]
    terms = list(set([term for topic_terms in proc_topics for term in topic_terms]))
    return terms

def filter_all():
    """Filter all the indexes based on their occurence in the search term
    """
    print('Loading inverted index...')
    with open("../data/INVERTED_INDEX_NOSTOP.pkl", "rb") as file:
        inverted_index = pickle.load(file)
    with open("../data/TOPIC_DICT_NOSTOP.pkl", "rb") as file:
        topic_dict = pickle.load(file)


    terms = split_stop_stem_topics(topic_dict)

    print('Writing filtered index...')

    filtered_inverted_index = filter_dict(inverted_index, terms)
    pickle.dump(filtered_inverted_index, open('../data/FILTERED_INVERTED_INDEX_NOSTOP.pkl', 'wb'))

if __name__ == "__main__":
    filter_all()
