import argparse
import os
from config import DATA_FOLDER
import random
import numpy as np
import yaml
import causaldag as cd
from tqdm import trange

random.seed(1738)
np.random.seed(1738)

parser = argparse.ArgumentParser(description='Sample from pairs of DAGs')
parser.add_argument('--folder', type=str, help='Folder to read DAGs from')
parser.add_argument('--nsamples', '-n', type=int, help='Number of samples')

args = parser.parse_args()

dataset_folder = os.path.join(DATA_FOLDER, args.folder)
dataset_config = yaml.load(open(os.path.join(dataset_folder, 'config.yaml')))

print('SAMPLING from DAGs')
for d in trange(dataset_config['npairs']):
    pair_folder = os.path.join(dataset_folder, 'pair%d' % d)
    parameters_folder = os.path.join(pair_folder, 'parameters')
    B1 = np.loadtxt(os.path.join(parameters_folder, 'B1.txt'))
    B2 = np.loadtxt(os.path.join(parameters_folder, 'B2.txt'))
    sigmas1 = np.loadtxt(os.path.join(parameters_folder, 'Sigma1.txt'))
    sigmas2 = np.loadtxt(os.path.join(parameters_folder, 'Sigma2.txt'))

    dag1 = cd.GaussDAG.from_amat(B1, variances=sigmas1)
    dag2 = cd.GaussDAG.from_amat(B2, variances=sigmas2)
    X1 = dag1.sample(args.nsamples)
    X2 = dag2.sample(args.nsamples)
    S1 = (X1.T @ X1)/args.nsamples
    S2 = (X2.T @ X2)/args.nsamples
    K1 = np.linalg.inv(S1)
    K2 = np.linalg.inv(S2)

    samples_folder = os.path.join(pair_folder, 'samples_n=%d' % args.nsamples)
    os.makedirs(samples_folder)
    np.savetxt(os.path.join(samples_folder, 'X1.txt'), X1)
    np.savetxt(os.path.join(samples_folder, 'X2.txt'), X2)
    np.savetxt(os.path.join(samples_folder, 'S1.txt'), S1)
    np.savetxt(os.path.join(samples_folder, 'S2.txt'), S2)
    np.savetxt(os.path.join(samples_folder, 'K1.txt'), K1)
    np.savetxt(os.path.join(samples_folder, 'K2.txt'), K2)



