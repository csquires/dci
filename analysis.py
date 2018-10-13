import xarray as xr
import numpy as np
from config import DATA_FOLDER
import os
import yaml
import itertools as itr


def edge2str(e):
    i, j = e
    return '%s,%s' % (i, j)


def get_edges_da(folder, algs, nsamples_list, alphas):
    dataset_folder = os.path.join(DATA_FOLDER, folder)
    dataset_config = yaml.load(open(os.path.join(dataset_folder, 'config.yaml')))
    npairs = dataset_config['npairs']
    nnodes = dataset_config['nnodes']
    possible_edges = list(itr.permutations(range(nnodes), 2))

    edges_da = xr.DataArray(
        np.zeros([npairs, len(algs) + 1, len(possible_edges), len(nsamples_list), len(alphas)], dtype=bool),
        dims=['pair', 'alg', 'edge', 'nsamples', 'alpha'],
        coords={
            'pair': list(range(npairs)),
            'alg': algs + ['true'],
            'edge': list(map(edge2str, possible_edges)),
            'nsamples': nsamples_list,
            'alpha': alphas
        }
    )
    for d, nsamples in itr.product(range(npairs), nsamples_list):
        pair_folder = os.path.join(dataset_folder, 'pair%d' % d)
        true_B1 = np.loadtxt(os.path.join(pair_folder, 'parameters', 'B1.txt'))
        true_B2 = np.loadtxt(os.path.join(pair_folder, 'parameters', 'B2.txt'))
        for i, j in possible_edges:
            if (true_B1[i, j] == 0) != (true_B2[i, j] == 0):
                edges_da.loc[dict(pair=d, alg='true', edge=edge2str((i, j)), nsamples=nsamples)] = True

        for alg in algs:
            results_folder = os.path.join(pair_folder, 'samples_n=%d' % nsamples, 'results', alg)
            if alg == 'dci':
                est_ddags = yaml.load(open(os.path.join(results_folder, 'estimated_ddags.yaml')))
                for alpha in alphas:
                    for (i, j) in est_ddags[alpha]:
                        edges_da.loc[dict(pair=d, alg=alg, edge=edge2str((i, j)), nsamples=nsamples, alpha=alpha)] = True
            elif alg == 'pcalg':
                for alpha in alphas:
                    alpha_folder = os.path.join(results_folder, 'alpha={:.2e}'.format(alpha))
                    est_B1 = np.loadtxt(os.path.join(alpha_folder, 'A1.txt'))
                    est_B2 = np.loadtxt(os.path.join(alpha_folder, 'A2.txt'))
                    for i, j in possible_edges:
                        if (est_B1[i, j] == 0) != (est_B2[i, j] == 0):
                            edges_da.loc[dict(pair=d, alg=alg, edge=edge2str((i, j)), nsamples=nsamples, alpha=alpha)] = True
            else:
                est_B1 = np.loadtxt(os.path.join(results_folder, 'A1.txt'))
                est_B2 = np.loadtxt(os.path.join(results_folder, 'A2.txt'))
                for i, j in possible_edges:
                    if (est_B1[i, j] == 0) != (est_B2[i, j] == 0):
                        edges_da.loc[dict(pair=d, alg=alg, edge=edge2str((i, j)), nsamples=nsamples, alpha=alpha)] = True

    return edges_da


def edges_da2_exact_recovery(edges_da):
    algs = set(edges_da.coords['algs']) - {'true'}
    nsamples_list = edges_da.coords['nsamples']
    alphas = edges_da.coords['alpha']
    exact_recovery_da = xr.DataArray(
        np.zeros([len(algs), len(nsamples_list), len(alphas)]),
        dims=['alg', 'nsamples', 'alpha'],
        coords={
            'alg': algs,
            'nsamples': nsamples_list,
            'alpha': alphas
        }
    )

    return exact_recovery_da


if __name__ == '__main__':
    e = get_edges_da('fig1_data', ['pcalg', 'ges', 'dci'], [1000], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])



