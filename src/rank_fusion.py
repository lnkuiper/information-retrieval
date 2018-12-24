"""Fuses two rankings into one, using various methods

Attributes:
    OUTPUT_DIR (str): Directory where rankings are stored
    RESULT_DIR (str): Directory where trec_eval results are stored
"""
import numpy as np

from time import time
from os import remove, system
from re import findall
from subprocess import Popen, PIPE

RESULT_DIR = '../results/'
OUTPUT_DIR = '../outputs/'


def get_results(results_file):
    """Extracts MAP and P30 scores from trec_eval output
    
    Args:
        results_file (str): Path to file containing trec_eval output
    
    Returns:
        dict: Scores
    """
    with open(results_file, 'r') as file_handle:
        content = file_handle.read()
    results = findall('[0-1]\.[0-9]*', content)
    names = [findall('[^\s]*', content)[0], findall('\n[^\s]*', content)[0][1:]]
    result_dict = {names[0]:float(results[0]),names[1]:float(results[1])}
    return result_dict


def ranking_list_to_dict(ranking_list):
    """Makes a dictionary of a result file for min-rank and borda fusion
    
    Args:
        ranking_list (list of str): lines in trec_eval input file
    
    Returns:
        dict: topic - [(rank, doc_id, score)]
    """
    ranking_list = [str(ranking).split(' ') for ranking in ranking_list]
    # topic num, rank, doc_id, score
    ranking_list = [(ranking[0], ranking[3], ranking[2], ranking[4]) for ranking in ranking_list]
    result_dict = {}
    for topic_num, rank, doc_id, score in ranking_list:
        if result_dict.get(topic_num):
            result_dict[topic_num].append((rank, doc_id, score))
        else:
            result_dict[topic_num] = [(rank, doc_id, score)]
    return result_dict


def ranking_list_to_dict_interp(ranking_list):
    """Makes a dictionary of a result file for interpolation fusion
    
    Args:
        ranking_list (list of str): lines in trec_eval input file
    
    Returns:
        dict: topic - [(rank, doc_id, score)]
    """
    ranking_list = [str(ranking).split(' ') for ranking in ranking_list]
    # topic num, score, doc_id
    ranking_list = [(ranking[0], ranking[4], ranking[2]) for ranking in ranking_list]
    result_dict = {}
    for topic_num, score, doc_id in ranking_list:
        if result_dict.get(topic_num):
            result_dict[topic_num].append((score, doc_id))
        else:
            result_dict[topic_num] = [(score, doc_id)]
    return result_dict


def interp_array(score_list):
    """Interpolates a list of ranking scores
    
    Args:
        score_list (list of float): List of ranking scores
    
    Returns:
        list of float: Interpolated ranking scores
    """
    minimum = min(score_list)
    if minimum >= 0:
        score_list  = [score-minimum for score in score_list]
    else:
        score_list = [score+minimum for score in score_list]
    maximum = max(score_list)
    return [score/maximum for score in score_list]


def min_fuse_rank_tuples(ranking_tuples1, ranking_tuples2):
    """Min-rank fusion of two rankings
    
    Args:
        ranking_tuples1 (list of (rank, doc_id, score)): First ranking
        ranking_tuples2 (list of (rank, doc_id, score)): Second ranking
    
    Returns:
        list of (rank, doc_id: fused ranking of first and second ranking
    """
    fused_rank_tuples = []
    parsed_docs = []
    for tuple1, tuple2 in zip(ranking_tuples1, ranking_tuples2):
        rank1, doc_id1, _  = tuple1
        rank2, doc_id2, _ = tuple2
        if doc_id1 not in parsed_docs:
            fused_rank_tuples.append((len(fused_rank_tuples), doc_id1))
            parsed_docs.append(doc_id1)
        if doc_id2 not in parsed_docs:
            fused_rank_tuples.append((len(fused_rank_tuples), doc_id2))
            parsed_docs.append(doc_id2)
    return fused_rank_tuples[:len(ranking_tuples1)]


def borda_fuse_rank_tuples(ranking_tuples1, ranking_tuples2):
    """borda-rank fusion of two rankings
    
    Args:
        ranking_tuples1 (list of (rank, doc_id, score)): First ranking
        ranking_tuples2 (list of (rank, doc_id, score): Second ranking
    
    Returns:
        list of (rank, doc_id): fused ranking of first and second ranking
    """
    # Unpack
    scores1 = [len(ranking_tuples1)-int(r[0]) for r in ranking_tuples1]
    scores2 = [len(ranking_tuples2)-int(r[0]) for r in ranking_tuples2]
    docs1 = [r[1] for r in ranking_tuples1]
    docs2 = [r[1] for r in ranking_tuples2]


    #Fusing two score to one document id
    fused_scores = list(zip(docs1, scores1))
    for document2, score2 in list(zip(docs2, scores2)):
        found = False
        for document1, score1 in fused_scores:
            if document2 == document1:
                score1 += score2
                found = True
        if not found:
            fused_scores.append((document2,score2))

    sorted_scores = sorted(fused_scores, key=lambda tup: -tup[1])
    sorted_docs = [tupl[0] for tupl in sorted_scores][:1000]
    return list(zip(range(0,1000),sorted_docs))


def interp_fuse_rank_tuples(ranking_tuples1, ranking_tuples2):
    """Interp-rank fusion of two rankings
    
    Args:
        ranking_tuples1 (list of (rank, doc_id, score)): First ranking
        ranking_tuples2 (list of (rank, doc_id, score): Second ranking
    
    Returns:
        list of (rank, doc_id): fused ranking of first and second ranking
    """

    # Unpack
    scores1 = [float(r[2]) for r in ranking_tuples1]
    scores2 = [float(r[2]) for r in ranking_tuples2]
    docs1 = [r[1] for r in ranking_tuples1]
    docs2 = [r[1] for r in ranking_tuples2]

    # Interpolate
    scores1 = interp_array(scores1)
    scores2 = interp_array(scores2)

    #Fusing two score to one document id
    fused_scores = list(zip(docs1, scores1))
    for document2, score2 in list(zip(docs2, scores2)):
        found = False
        for document1, score1 in fused_scores:
            if document2 == document1:
                score1 += score2
                found = True
        if not found:
            fused_scores.append((document2,score2))

    sorted_scores = sorted(fused_scores, key=lambda tup: -tup[1])
    sorted_docs = [tupl[0] for tupl in sorted_scores][:1000]
    return list(zip(range(0,1000),sorted_docs))


def rank_fusion(fname1, fname2, fuse_type, out_file_name):
    """Fuses ranks and puts them into an output file for trec_eval
    
    Args:
        fname1 (str): File name of trec_eval input of first ranking
        fname2 (str): File name of trec_eval input of second ranking
        fuse_type (str): One of 'min', 'interp', 'borda'
        out_file_name (str): Where to store output
    
    Returns:
        dict: Scores
    """
    with open(fname1, 'r') as file:
        ranking_list1 = file.readlines()
    with open(fname2, 'r') as file:
        ranking_list2 = file.readlines()
    ranking_dict1 = ranking_list_to_dict(ranking_list1)
    ranking_dict2 = ranking_list_to_dict(ranking_list2)

    trec_eval_out = []
    end_str = ' STANDARD'
    for topic in sorted(ranking_dict1):
        if fuse_type == 'min':
            fused_ranking = min_fuse_rank_tuples(ranking_dict1.get(topic), ranking_dict2.get(topic))
        elif fuse_type == 'interp':
            fused_ranking = interp_fuse_rank_tuples(ranking_dict1.get(topic), ranking_dict2.get(topic))
        elif fuse_type == 'borda':
            fused_ranking = borda_fuse_rank_tuples(ranking_dict1.get(topic), ranking_dict2.get(topic))
        else:
            print('Selected ranking method does not exist!')
        start_str = topic + ' Q0 '
        for rank, doc_id in fused_ranking:
            trec_eval_line = start_str + ' '.join([doc_id, str(rank), str(1/int(rank+1))]) + end_str
            trec_eval_out.append(trec_eval_line)

    with open(out_file_name, 'w+') as file:
        for line in trec_eval_out:
            print(line, file=file)
    results = get_results(out_file_name)


    system('./../trec_eval.9.0/trec_eval -m map -m P.30 ../data/qrels ' + out_file_name + ' > ' + './temp_score')

    results = get_results('./temp_score') 
    remove('./temp_score')
   
    remove(out_file_name)
    return results


def main():
    """Summary
    """
    None


if __name__ == '__main__':
    main()