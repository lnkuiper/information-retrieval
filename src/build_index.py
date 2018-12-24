"""Methods used for building indexes for TREC data
   this is an ad-hoc approach, index was build once and shared via USB
   
Attributes:
    DOC_FREQ_INDEX (dict): term - doc_freq
    DOC_LEN_INDEX (dict): doc_id - doc_len
    FB_DIR (str): directory of FB files
    FB_FILE_NAMES (TYPE): file names in FB directory
    FR_DIR (str): directory of FR files
    FR_FILE_NAMES (TYPE): Descriptiofile names in FR directory
    FT_DIR (str): directory of FT files
    FT_FILE_NAMES (TYPE): file names in FR directory
    INVERTED_INDEX (dict): term - postings
    LA_DIR (str): directory of LA files
    LA_FILE_NAMES (TYPE): file names in LA directory
"""
import os
import _pickle as pickle
import re

from tqdm import tqdm

import text_manipulation as tm
import filter_indexes as fi

FB_DIR = "../data/TREC_VOL_5/fbis/"
FR_DIR = "../data/TREC_VOL_4/fr94/"
FT_DIR = "../data/TREC_VOL_4/ft/"
LA_DIR = "../data/TREC_VOL_5/latimes/"

FB_FILE_NAMES = os.listdir(FB_DIR)[:-2]
FR_FILE_NAMES = os.listdir(FR_DIR)[13:-2]
FT_FILE_NAMES = [fn for fn in os.listdir(FT_DIR) if fn.endswith('.txt')]
LA_FILE_NAMES = os.listdir(LA_DIR)[:-2]

INVERTED_INDEX = {}
DOC_LEN_INDEX = {}
DOC_FREQ_INDEX = {}

def process_dir(data_dir, file_names, extra_space=1):
    """Processes a directory of files containing documents
    
    Args:
        data_dir (str): data directory containing TREC data
        file_names (list of str): file names to parse in data_dir
        extra_space (bool, optional): used by tm.get_blocks
    """
    doc_ids = []
    doc_texts = []

    print('Reading files...')
    documents = []
    for file_name in tqdm(file_names):
        file_path = data_dir + file_name

        with open(file_path, 'r', encoding='latin-1') as file:
            data = ''.join(file.readlines())
            documents += tm.get_blocks(data, 'DOC')

    print('Reading complete. Processing...')
    for doc in tqdm(documents):
        text = tm.get_blocks(doc, 'TEXT', extra_space=0)
        if text:
            doc_ids.append(tm.get_blocks(doc, 'DOCNO', extra_space=extra_space)[0])
            text_lines = text[0].split('\n')
            text = '\n'.join([line for line in text_lines if not line.startswith('<')])
            doc_texts.append(text)

    print('Processing complete. Indexing...')
    for doc_id, doc_text in tqdm(list(zip(doc_ids, doc_texts))):
        add_to_indexes(doc_id, doc_text)
    print('Indexing complete.')

def create_topic_dict():
    """Creates a dictionary of number: topic from TREC topics file
    """
    with open('../data/topics') as file:
        topic_text = file.read()

    num_matches = [num.strip() for num in re.findall(r'<num> Number:([\s\S]*?)<title>', topic_text)]
    topic_matches = [topic.strip() for topic in re.findall(r'<title>([\s\S]*?)<desc>', topic_text)]

    topic_dict = {}
    for i, num_match in enumerate(num_matches):
        topic_dict[num_match] = topic_matches[i]

    pickle.dump(topic_dict, open('../data/TOPIC_DICT_NOSTOP.pkl', 'wb'))

def add_to_indexes(doc_id, doc_text):
    """Adds a document to the indexes
    
    Args:
        doc_id (str): Identifier of document
        doc_text (str): Text of document
    """
    bag_of_words = tm.process_text(doc_text)
    DOC_LEN_INDEX[doc_id] = len(bag_of_words)

    term_freq_pairs = tm.to_frequency_pairs(bag_of_words)
    for term, freq in term_freq_pairs:
        if INVERTED_INDEX.get(term):
            INVERTED_INDEX[term].append((doc_id, freq))
        else:
            INVERTED_INDEX[term] = [(doc_id, freq)]

def main():
    process_dir(FB_DIR, FB_FILE_NAMES)
    process_dir(FR_DIR, FR_FILE_NAMES)
    process_dir(FT_DIR, FT_FILE_NAMES, extra_space=0)
    process_dir(LA_DIR, LA_FILE_NAMES)

    AVERAGE_DOC_LENGTH = 0
    DOC_COUNT = 0
    for doc_id_key in DOC_LEN_INDEX:
        AVERAGE_DOC_LENGTH += DOC_LEN_INDEX.get(doc_id_key)
        DOC_COUNT += 1
    AVERAGE_DOC_LENGTH = AVERAGE_DOC_LENGTH / DOC_COUNT

    for term_key in INVERTED_INDEX:
        INVERTED_INDEX[term_key] = sorted(INVERTED_INDEX[term_key])
        DOC_FREQ_INDEX[term_key] = len(INVERTED_INDEX[term_key])

    print('Saving indices...')

    pickle.dump(INVERTED_INDEX, open('../data/INVERTED_INDEX_NOSTOP.pkl', 'wb'))
    pickle.dump(DOC_LEN_INDEX, open('../data/DOC_LEN_INDEX_NOSTOP.pkl', 'wb'))
    pickle.dump(DOC_FREQ_INDEX, open('../data/DOC_FREQ_INDEX_NOSTOP.pkl', 'wb'))

    create_topic_dict()
    print('Saving complete.')

    fi.filter_all()

if __name__ == "__main__":
    main()