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
    algs = list(set(edges_da.coords['alg'].values) - {'true'})
    nsamples_list = list(edges_da.coords['nsamples'].values)
    alphas = list(edges_da.coords['alpha'].values)
    exact_recovery_da = xr.DataArray(
        np.zeros([len(algs), len(nsamples_list), len(alphas)]),
        dims=['alg', 'nsamples', 'alpha'],
        coords={
            'alg': algs,
            'nsamples': nsamples_list,
            'alpha': alphas
        }
    )
    for alg, nsamples, alpha in itr.product(algs, nsamples_list, alphas):
        true_edges_by_dag = edges_da.sel(alg='true', nsamples=nsamples, alpha=alpha)
        est_edges_by_dag = edges_da.sel(alg=alg, nsamples=nsamples, alpha=alpha)
        matching_dags = (true_edges_by_dag == est_edges_by_dag).all(dim='edge')
        exact_recovery_da.loc[dict(alg=alg, nsamples=nsamples, alpha=alpha)] = matching_dags.sum() / len(matching_dags)

    return exact_recovery_da


def edge_da2tpr_fpr(edges_da):
    algs = list(set(edges_da.coords['alg'].values) - {'true'})
    nsamples_list = list(edges_da.coords['nsamples'].values)
    alphas = list(edges_da.coords['alpha'].values)
    pair_ixs = edges_da.coords['pair'].values

    rates_da = xr.DataArray(
        np.zeros([len(pair_ixs), len(algs), len(nsamples_list), len(alphas), 2]),
        dims=['pair', 'alg', 'nsamples', 'alpha', 'rate'],
        coords={
            'pair': pair_ixs,
            'alg': algs,
            'nsamples': nsamples_list,
            'alpha': alphas,
            'rate': ['tpr', 'fpr']
        }
    )
    for alg, nsamples, alpha in itr.product(algs, nsamples_list, alphas):
        # get set of true edges and true non-edges for each pair
        true_edges_by_pair = edges_da.sel(alg='true', nsamples=nsamples, alpha=alpha)
        true_edges_by_pair = [set(np.where(true_edges_by_pair.sel(pair=pair_ix))[0]) for pair_ix in pair_ixs]
        true_nonedges_by_pair = [set(edges_da.coords['pair'].values) - true_edges for true_edges in true_edges_by_pair]

        # get set of estimated edges and estimated non-edges for each pair
        est_edges_by_pair = edges_da.sel(alg=alg, nsamples=nsamples, alpha=alpha)
        est_edges_by_pair = [set(np.where(est_edges_by_pair.sel(pair=pair_ix))[0]) for pair_ix in pair_ixs]
        est_nonedges_by_pair = [set(edges_da.coords['pair'].values) - est_edges for est_edges in est_edges_by_pair]

        # get false positives and true positives for each pair
        false_positives_by_pair = [est_edges_by_pair[pair_ix] - true_edges_by_pair[pair_ix] for pair_ix in pair_ixs]
        true_positives_by_pair = [est_edges_by_pair[pair_ix] & true_edges_by_pair[pair_ix] for pair_ix in pair_ixs]

        num_fp_by_pair = np.array([len(fps) for fps in false_positives_by_pair])
        num_tp_by_pair = np.array([len(tps) for tps in true_positives_by_pair])
        num_negatives_by_pair = np.array([len(negatives) for negatives in true_nonedges_by_pair])
        num_positives_by_pair = np.array([len(positives) for positives in true_edges_by_pair])
        rates_da.loc[dict(alg=alg, nsamples=nsamples, alpha=alpha, rate='fpr')] = num_fp_by_pair / num_negatives_by_pair
        rates_da.loc[dict(alg=alg, nsamples=nsamples, alpha=alpha, rate='tpr')] = num_tp_by_pair / num_positives_by_pair

    return rates_da


if __name__ == '__main__':
    import pdb
    e = get_edges_da('fig1_data', ['pcalg', 'ges', 'dci'], [1000], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    exact_recovery = edges_da2_exact_recovery(e)
    rates_da = edge_da2tpr_fpr(e)




