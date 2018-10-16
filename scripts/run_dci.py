import argparse
import os
from config import DATA_FOLDER
import yaml
from multiprocessing import Pool, cpu_count

import random
import numpy as np
from dci import estimate_dug, estimate_ddag, estimate_ddag_skeleton
from utils.sys_utils import listify_dict
import itertools as itr
from tqdm import tqdm

random.seed(1729)
np.random.seed(1729)


def estimate_ddag_skeleton_(tup): return estimate_ddag_skeleton(*tup)


def estimate_ddag_(tup): return estimate_ddag(*tup)


def estimate_dug_(tup): return estimate_dug(*tup)


def kliep_array_to_edges(kliep_array, p, thresh=0.):
    edges = set()
    k = 0
    for j in range(1, p):
        for i in range(j):
            if kliep_array[k] > thresh:
                edges.add((i, j))
            k += 1
    return edges


parser = argparse.ArgumentParser(description='Run DCI on a data set')
parser.add_argument('--folder', type=str, help='Folder name of data set')
parser.add_argument('--nsamples', type=int, help='Number of samples to use in analyzing results')
parser.add_argument('--alphas2', type=float, nargs='+', help='Significance thresholds used in estimating D-DAGs')
parser.add_argument('--dug', type=str, default='constraint', help='Method for estimating the D-UG')
parser.add_argument(
    '--alpha1',
    type=float,
    nargs='?',
    default=.001,
    help='Significance threshold if D-UG estimation method is "constraint"'
)
parser.add_argument(
    '--lambda-kliep',
    type=float,
    nargs='?',
    default=4,
    help='Sparsity parameter if the D-UG estimation method is "kliep"'
)
parser.add_argument(
    '--threshold-kliep',
    type=float,
    nargs='?',
    default=.05,
    help='Threshold on edge weights if the D-UG estimation method is "kliep"'
)
args = parser.parse_args()

dataset_folder = os.path.join(DATA_FOLDER, args.folder)
dataset_config = yaml.load(open(os.path.join(dataset_folder, 'config.yaml')))
npairs = dataset_config['npairs']
nnodes = dataset_config['nnodes']

pair_folders = [os.path.join(dataset_folder, 'pair%d' % d) for d in range(npairs)]
samples_folders = [os.path.join(pair_folder, 'samples_n=%d' % args.nsamples) for pair_folder in pair_folders]

if args.dug == 'constraint':
    alg_suffix = 'dci-c'
elif args.dug == 'fc':
    alg_suffix = 'dci-fc'
elif args.dug == 'kliep':
    alg_suffix = 'dci-k'
else:
    raise ValueError('dug must be "constraint", "kliep", or "fc"')
results_folders = [os.path.join(sample_folder, 'results', alg_suffix) for sample_folder in samples_folders]
for results_folder in results_folders:
    os.makedirs(results_folder, exist_ok=True)

print('LOADING DATA')
X1s = [np.loadtxt(os.path.join(folder, 'X1.txt')) for folder in samples_folders]
X2s = [np.loadtxt(os.path.join(folder, 'X2.txt')) for folder in samples_folders]
K1s = [np.loadtxt(os.path.join(folder, 'K1.txt')) for folder in samples_folders]
K2s = [np.loadtxt(os.path.join(folder, 'K2.txt')) for folder in samples_folders]

with Pool(cpu_count()-1) as p:
    print('FINDING D-UGs')
    if args.dug == 'constraint':
        tups = zip(K1s, K2s, [args.nsamples]*npairs, [args.nsamples]*npairs, [args.alpha1]*npairs)
        dug_results = list(tqdm(p.imap(estimate_dug_, tups), total=npairs))
        est_dugs, est_changed_nodes = zip(*dug_results)
    elif args.dug == 'kliep':
        dug_filenames = [
            os.path.join(folder, 'results', 'kliep', 'lambda=%.3f' % args.lambda_kliep, 'K.txt')
            for folder in samples_folders
        ]
        kliep_arrays = [np.loadtxt(fn) for fn in dug_filenames]
        est_dugs = [kliep_array_to_edges(kliep_array, nnodes, thresh=args.threshold_kliep) for kliep_array in kliep_arrays]
        est_changed_nodes = [{i for i, j in edges} | {j for i, j in edges} for edges in est_dugs]
    else:
        est_dugs = [set(itr.combinations(range(nnodes), 2))]*npairs
        est_changed_nodes = [set(range(nnodes))]*npairs

    print('FINDING DDAG SKELETONS')
    tups = zip(X1s, X2s, est_dugs, est_changed_nodes, [args.alphas2]*npairs)
    ddag_skeleton_results = list(tqdm(p.imap(estimate_ddag_skeleton_, tups), total=npairs))
    ddag_skeletons, _ = zip(*ddag_skeleton_results)
    for skel, results_folder in zip(ddag_skeletons, results_folders):
        yaml.dump(listify_dict(skel), open(os.path.join(results_folder, 'estimated_ddag_skeletons.yaml'), 'w'))

    print('FINDING DDAGS')
    estimated_ddags_by_pair = [{} for _ in range(npairs)]
    for alpha2 in args.alphas2:
        tups = zip(X1s, X2s, [skel[alpha2] for skel in ddag_skeletons], est_changed_nodes, [alpha2]*npairs)
        ddags = list(tqdm(p.imap(estimate_ddag_, tups), total=npairs))
        for d, ddag in enumerate(ddags):
            estimated_ddags_by_pair[d][alpha2] = ddag

    for ddags, results_folder in zip(estimated_ddags_by_pair, results_folders):
        yaml.dump(listify_dict(ddags), open(os.path.join(results_folder, 'estimated_ddags.yaml'), 'w'))
