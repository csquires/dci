import argparse
import os
from config import DATA_FOLDER
from utils import math_utils
import yaml
from tqdm import trange

import random
import numpy as np
random.seed(1738)
np.random.seed(1738)

parser = argparse.ArgumentParser(description='Create pairs of DAGs')
parser.add_argument('--npairs', '-d', type=int, help='Number of DAG pairs to create')
parser.add_argument('--folder', type=str, help='Folder name to save results into')

# Parameters for DAG
parser.add_argument('--nnodes', '-p', type=int, help='Number of nodes in each DAG')
parser.add_argument('--nneighbors', '-s', type=float, help='Expected neighborhood size in original DAG')

# Parameters for changes between DAGs
parser.add_argument(
    '--changes',
    type=str,
    help='Method of generating second DAG from the first. "fixed" or "independent". See code for details.'
)
parser.add_argument(
    '--percent-added',
    type=float,
    nargs='?',
    default=0,
    help='Percent edges added. Exact effect depends on value of the "--changes" argument'
)
parser.add_argument(
    '--percent-removed',
    type=float,
    nargs='?',
    default=0,
    help='Percent edges removed. Exact effect depends on value of the "--changes" argument'
)
parser.add_argument(
    '--percent-changed',
    type=float,
    nargs='?',
    default=0,
    help='Percent edges changed. Exact effect depends on value of the "--changes" argument'
)
parser.add_argument(
    '--changed-variances',
    type=int,
    nargs='?',
    default=0,
    help='Number of nodes which change their variance'
)


parser.add_argument('--nsamples', '-n', type=int, help='Number of samples from each pair of DAGs')

args = parser.parse_args()
if args.changes not in ['fixed', 'independent']:
    raise ValueError('The value of the "changes" parameter must be either "fixed" or "independent"')

dataset_folder = os.path.join(DATA_FOLDER, args.folder)
os.makedirs(dataset_folder, exist_ok=True)
yaml.dump(vars(args), open(os.path.join(dataset_folder, 'config.yaml'), 'w'))
print('CREATING DAGs')
for d in trange(args.npairs):
    B1 = math_utils.random_dag(args.nnodes, args.nneighbors/(args.nnodes-1))
    B2, _ = math_utils.random_dag_changes(
        B1,
        r=args.percent_removed,
        a=args.percent_added,
        c=args.percent_changed,
        type_=args.changes
    )
    sigma1s = np.ones(args.nnodes)
    sigma2s = math_utils.random_variances(args.nnodes, args.changed_variances)

    pair_folder = os.path.join(dataset_folder, 'pair%s' % d)
    os.makedirs(pair_folder)
    parameters_folder = os.path.join(pair_folder, 'parameters')
    os.makedirs(parameters_folder)
    np.savetxt(os.path.join(parameters_folder, 'B1.txt'), B1)
    np.savetxt(os.path.join(parameters_folder, 'B2.txt'), B2)
    np.savetxt(os.path.join(parameters_folder, 'Sigma1.txt'), sigma1s)
    np.savetxt(os.path.join(parameters_folder, 'Sigma2.txt'), sigma2s)



