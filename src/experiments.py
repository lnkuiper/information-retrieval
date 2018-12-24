"""Automated retrieval experiments
"""
import os
import numpy as np
from re import findall
from ranking import ql, bm25, rank_evaluate
from rank_fusion import rank_fusion

def experiment_ql_jm():
    """Experiments query likelihood retrieval with Jelinek-Mercer smoothing,
    lambda coefficient increments of 0.025 at a time, 40 experiments
    """
    for l in np.linspace(0.025, 1, 40):
        argdict = {
            'k': 1000,
            'fun': ql,
            'smoothing': 'jm',
            'lambda': l
        }
        rank_evaluate(argdict=argdict)


def experiment_ql_dir():
    """Experiments query likelihood retrieval with Dirichlet smoothing,
    mu prior increments by 100 at a time, 40 experiments
    """
    for mu in np.linspace(100, 4000, 40):
        argdict = {
            'k': 1000,
            'fun': ql,
            'smoothing': 'dir',
            'mu': mu
        }
        rank_evaluate(argdict=argdict)


def experiment_bm():
    """Experiments Okapi BM25 retrieval with k and b parameters,
    both 21 different values, 20*16 = 320 experiments
    """
    for b in np.linspace(0.05, 1, 20):
        for k1 in np.linspace(0.125, 2, 16):
            argdict = {
                'k': 1000,
                'fun': bm25,
                'k1': k1,
                'b': b
            }
            rank_evaluate(argdict=argdict)


def run_eval():
    """Runs trec_eval, stores results
    """
    output_dir = '../outputs/'
    result_dir = '../results/'
    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            with open(result_dir + filename, 'w+') as file:
                os.system('./../trec_eval.9.0/trec_eval -m map -m P.30 ../data/qrels ' + output_dir + filename + ' > ' + result_dir + filename)


def get_results_from_dir(result_directory):
    """Gets all results from files containing trec_eval output in a directory
    
    Args:
        result_directory (str): Path to directory
    
    Returns:
        dict: scores
    """
    result_dict= {}
    for filename in sorted(os.listdir(result_directory)):
        result = get_results(result_directory + filename)
        for key,value in result.items():
            if key not in result_dict.keys():
                result_dict[key] = [value]
            else:
                result_dict[key].append(value)
        file_args = filename.rsplit('.',1)[0].split('_')
        while isfloat(file_args[-1]):
            if file_args[-2] not in result_dict.keys():
                result_dict[file_args[-2]] = [float(file_args[-1])]
            else:
                result_dict[file_args[-2]].append(float(file_args[-1]))
            file_args = file_args[:-2]

    return result_dict


def get_results(results_file):
    """Extracts MAP and P30 scores from trec_eval output
    
    Args:
        results_file (str): Path to file containing trec_eval output
    
    Returns:
        dict: scores
    """
    with open(results_file, 'r') as file_handle:
        content = file_handle.read()
    results = findall('[0-1]\.[0-9]*', content)
    names = [findall('[^\s]*', content)[0], findall('\n[^\s]*', content)[0][1:]]
    result_dict = {names[0]:float(results[0]),names[1]:float(results[1])}
    return result_dict


def fusion_on_folder(fuse_folder):
    """Calls rank fusion methods on ranking stored in folder
    
    Args:
        fuse_folder (str): Folder name where rankings are stored
    """
    for folder in sorted(os.listdir(fuse_folder)):
        print(folder)
        filenames = sorted(os.listdir(fuse_folder+folder))
        print(' - Minimal fusion:')
        for i in range(len(filenames)-1):
            for j in range(i+1,len(filenames)):
                print('\t'+filenames[i] + ' + ' + filenames[j])
                print('\t'+str(rank_fusion(fuse_folder+folder+'/'+filenames[i],fuse_folder+folder+'/'+filenames[j],'min','temporary_results')[folder]))
        print(' - Interpolated fusion:')
        for i in range(len(filenames)-1):
            for j in range(i+1,len(filenames)):
                print('\t'+filenames[i] + ' + ' + filenames[j])

                print('\t'+str(rank_fusion(fuse_folder+folder+'/'+filenames[i],fuse_folder+folder+'/'+filenames[j],'interp','temporary_results')[folder]))
        print(' - Borda count:')
        for i in range(len(filenames)-1):
            for j in range(i+1,len(filenames)):
                print('\t'+filenames[i] + ' + ' + filenames[j])

                print('\t'+str(rank_fusion(fuse_folder+folder+'/'+filenames[i],fuse_folder+folder+'/'+filenames[j],'borda','temporary_results')[folder]))

        print('-----------------------')




def isfloat(value):
    """Type check, used for MAP and P30 score parsing
    
    Args:
        value (any): object to be checked
    
    Returns:
        Bool: Whether object is float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def main():
    """Executes experiments
    """
    experiment_ql_jm()
    experiment_ql_dir()
    experiment_bm()
    run_eval()
    # fusion_on_folder('../outputs/best/')


if __name__ == '__main__':
    main()
