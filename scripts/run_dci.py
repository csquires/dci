import argparse
import os
from config import DATA_FOLDER
import yaml
from multiprocessing import Pool, cpu_count

import random
import numpy as np
from dci import estimate_dug, estimate_ddag, estimate_ddag_skeleton
from utils.sys_utils import listify_dict

random.seed(1729)
np.random.seed(1729)

parser = argparse.ArgumentParser(description='Run DCI on a data set')
parser.add_argument('--folder', type=str, help='Folder name of data set')
parser.add_argument('--nsamples', type=int, help='Number of samples to use in analyzing results')
parser.add_argument('--alpha1', type=float, nargs='?', default=.1, help='Significance threshold used in estimating D-UGs')
parser.add_argument('--alphas2', type=float, nargs='+', help='Significance thresholds used in estimating D-DAGs')

args = parser.parse_args()

dataset_folder = os.path.join(DATA_FOLDER, args.folder)
dataset_config = yaml.load(open(os.path.join(dataset_folder, 'config.yaml')))
npairs = dataset_config['npairs']

pair_folders = [os.path.join(dataset_folder, 'pair%d' % d) for d in range(npairs)]
samples_folders = [os.path.join(pair_folder, 'samples_n=%d' % args.nsamples) for pair_folder in pair_folders]
results_folders = [os.path.join(sample_folder, 'results', 'dci') for sample_folder in samples_folders]
for results_folder in results_folders:
    os.makedirs(results_folder)

X1s = [np.loadtxt(os.path.join(folder, 'X1.txt')) for folder in samples_folders]
X2s = [np.loadtxt(os.path.join(folder, 'X2.txt')) for folder in samples_folders]
K1s = [np.loadtxt(os.path.join(folder, 'K1.txt')) for folder in samples_folders]
K2s = [np.loadtxt(os.path.join(folder, 'K2.txt')) for folder in samples_folders]

with Pool(cpu_count()-1) as p:
    dug_results = p.starmap(estimate_dug, zip(K1s, K2s, [args.nsamples]*npairs, [args.nsamples]*npairs, [args.alpha1]*npairs))
    est_dugs, est_changed_nodes = zip(*dug_results)
    print('FOUND D-UGs')

    ddag_skeleton_results = p.starmap(estimate_ddag_skeleton, zip(X1s, X2s, est_dugs, est_changed_nodes, [args.alphas2]*npairs))
    ddag_skeletons, _ = zip(*ddag_skeleton_results)
    for skel, results_folder in zip(ddag_skeletons, results_folders):
        yaml.dump(listify_dict(skel), open(os.path.join(results_folder, 'estimated_ddag_skeletons.yaml'), 'w'))
    print('FOUND DDAG SKELETONS')

    estimated_ddags_by_pair = [{} for _ in range(npairs)]
    for alpha2 in args.alphas2:
        ddags = p.starmap(estimate_ddag, zip(X1s, X2s, [skel[alpha2] for skel in ddag_skeletons], est_changed_nodes, [alpha2]*npairs))
        for d, ddag in enumerate(ddags):
            estimated_ddags_by_pair[d][alpha2] = ddag
    print('FOUND DDAGs')

    for ddags, results_folder in zip(estimated_ddags_by_pair, results_folders):
        yaml.dump(listify_dict(ddags), open(os.path.join(results_folder, 'estimated_ddags.yaml'), 'w'))
