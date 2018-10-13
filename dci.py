import numpy as np
from utils import math_utils
from collections import Iterable
from scipy import stats
from sklearn import linear_model
import networkx as nx


def estimate_ddag(X1, X2, skeleton, changed_nodes, alpha=.05, max_set_size=None, verbose=False):
    """
    Infer the directions (when possible) of edges in the DAG difference skeleton

    :param X1: (n1 x p) data matrix from first context
    :param X2: (n2 x p) data matrix from second context
    :param skeleton: list of edges in the difference DAG skeleton to be oriented
    :param changed_nodes: set of nodes with at least one value changed from inf_K1 to inf_K2
    :param alpha: lower value => more likely to accept null hypothesis => more directed edges
    :param max_set_size
    :param verbose:
    :return:
    """
    if not isinstance(changed_nodes, set):
        raise TypeError('changed_nodes must be a set')

    n1, p1 = X1.shape
    n2, p2 = X2.shape

    S1 = X1.T @ X1
    S2 = X2.T @ X2

    printv = print if verbose else lambda x: None

    oriented_edges = set()
    for i, j in skeleton:
        printv("Attempting to orient edge (%d, %d)" % (i, j))
        not_i = changed_nodes - {i}
        not_j = changed_nodes - {j}

        X1_i = X1[:, i]
        X2_i = X2[:, i]
        X1_j = X1[:, j]
        X2_j = X2[:, j]
        powersets = zip(
            math_utils.powerset(not_i, max_set_size=max_set_size),
            math_utils.powerset(not_j, max_set_size=max_set_size)
        )
        for m_i, m_j in powersets:
            # print("m_i, m_j", m_i, m_j)
            # calculate regression coefficients
            K1_i = np.linalg.inv(S1[np.ix_(m_i, m_i)])
            K2_i = np.linalg.inv(S2[np.ix_(m_i, m_i)])
            b1_i = K1_i @ S1[i, m_i].T
            b2_i = K2_i @ S2[i, m_i].T

            # calculate p-value from sample residual variances
            X1_mi = X1[:, m_i]
            X2_mi = X2[:, m_i]
            ssr1_i = np.sum(np.square(X1_i - X1_mi @ b1_i.T))
            ssr2_i = np.sum(np.square(X2_i - X2_mi @ b2_i.T))
            var1_i = ssr1_i / (n1 - len(m_i))
            var2_i = ssr2_i / (n2 - len(m_i))
            p_i = math_utils.ftest_pvalue(var1_i, var2_i, n1 - len(m_i), n2 - len(m_i))
            if p_i > alpha:  # accept hypothesis that var1_i = var2_i, orient edge
                # print(var1_i, var2_i, p_i, alpha)
                edge = (j, i) if j in m_i else (i, j)
                oriented_edges.add(edge)
                printv("Oriented (%d, %d) as %s" % (i, j, edge))
                break

            # calculate regression coefficients
            K1_j = np.linalg.inv(S1[np.ix_(m_j, m_j)])
            K2_j = np.linalg.inv(S2[np.ix_(m_j, m_j)])
            b1_j = K1_j @ S1[j, m_j].T
            b2_j = K2_j @ S2[j, m_j].T

            # calculate p-value from sample residual variances
            X1_mj = X1[:, m_j]
            X2_mj = X2[:, m_j]
            ssr1_j = np.sum(np.square(X1_j - X1_mj @ b1_j.T))
            ssr2_j = np.sum(np.square(X2_j - X2_mj @ b2_j.T))
            var1_j = ssr1_j / (n1 - len(m_j))
            var2_j = ssr2_j / (n2 - len(m_j))
            p_j = math_utils.ftest_pvalue(var1_j, var2_j, n1 - len(m_j), n2 - len(m_j))
            if p_j > alpha:  # accept hypothesis that var1_j = var2_j, orient edge
                edge = (i, j) if i in m_j else (j, i)
                oriented_edges.add(edge)
                printv("Oriented (%d, %d) as %s" % (i, j, edge))
                break

    unoriented_edges = skeleton - oriented_edges - {(j, i) for i, j in oriented_edges}
    g = nx.DiGraph()
    for i, j in oriented_edges:
        g.add_edge(i, j)
    for i in g.nodes:
        for j in g.successors(i):
            if (i, j) in unoriented_edges:
                oriented_edges.add((i, j))
                unoriented_edges.remove((i, j))
            if (j, i) in unoriented_edges:
                oriented_edges.add((i, j))
                unoriented_edges.remove((j, i))

    return oriented_edges


def estimate_ddag_skeleton(X1, X2, candidate_edges, changed_nodes, alpha=.1, lasso_alpha=None, max_set_size=None, verbose=False):
    """

    :param X1: (n1 x p) data matrix from first context
    :param X2: (n2 x p) data matrix from second context
    :param candidate_edges: set of edges that could possibly be in the skeleton
    :param changed_nodes: nodes adjacent to a candidate edge or with changed variance
    :param alpha: significance level to reject null hypothesis b1 = b2. Lower alpha makes it
    easier to accept the null hypothesis, so more edges will be deleted.
    :param lasso_alpha:
    :param max_set_size:
    :param verbose:
    :return:
    """
    n1, p1 = X1.shape
    n2, p2 = X2.shape
    if p1 != p2:
        raise ValueError("X1 and X2 must have the same number of dimensions")
    if isinstance(alpha, Iterable):
        alpha_ = max(alpha)
    else:
        alpha_ = alpha

    retained_edges = set()
    retained_edges_with_p = {}
    deleted_edges = set()
    deleted_edges_with_p = {}

    candidate_edges = {tuple(sorted((i, j))) for i, j in candidate_edges}

    printv = print if verbose else (lambda x: 0)

    S1 = X1.T @ X1
    S2 = X2.T @ X2
    for i, j in candidate_edges:
        printv("Checking edge (%d, %d)" % (i, j))
        not_ij = changed_nodes - {i, j}
        is_regression_invariant = False
        max_p = float('-inf')
        X1_j = X1[:, j]
        X2_j = X2[:, j]
        X1_i = X1[:, i]
        X2_i = X2[:, i]
        for cond_set in math_utils.powerset(not_ij, max_set_size=max_set_size):
            m_i = cond_set + (i,)
            m_j = cond_set + (j,)
            X1_mi = X1[:, m_i]
            X2_mi = X2[:, m_i]
            X1_mj = X1[:, m_j]
            X2_mj = X2[:, m_j]

            # marginal precision matrices
            if lasso_alpha is None:
                K1_ij = np.linalg.inv(S1[np.ix_(m_i, m_i)])
                K2_ij = np.linalg.inv(S2[np.ix_(m_i, m_i)])
                K1_ji = np.linalg.inv(S1[np.ix_(m_j, m_j)])
                K2_ji = np.linalg.inv(S2[np.ix_(m_j, m_j)])
                b1_mij = K1_ij @ S1[j, m_i].T
                b2_mij = K2_ij @ S2[j, m_i].T
                b1_mji = K1_ji @ S1[i, m_j].T
                b2_mji = K2_ji @ S2[i, m_j].T
            else:
                clf = linear_model.Lasso(alpha=lasso_alpha)
                clf.fit(X1_mi, X1_j)
                b1_mij = clf.coef_
                clf.fit(X2_mi, X2_j)
                b2_mij = clf.coef_
                clf.fit(X1_mj, X1_i)
                b1_mji = clf.coef_
                clf.fit(X2_mj, X2_i)
                b2_mji = clf.coef_

            # calculate t_ij (regressing j on i) and find its p-value
            ssr1_ij = np.sum(np.square( X1_j - X1_mi @ b1_mij.T ))
            ssr2_ij = np.sum(np.square( X2_j - X2_mi @ b2_mij.T ))
            var1_ij = ssr1_ij / (n1 - len(m_i))
            var2_ij = ssr2_ij / (n2 - len(m_i))
            b1_ij = b1_mij[-1]
            b2_ij = b2_mij[-1]
            t_ij = (b1_ij - b2_ij)**2 * np.linalg.inv(var1_ij * K1_ij + var2_ij * K2_ij)[-1, -1]
            p_ij = 1 - stats.f.cdf(t_ij, 1, n1 + n2 - len(m_i) - len(m_j))
            if p_ij > alpha_:  # accept hypothesis that b1_ij = b2_ij, delete edge
                is_regression_invariant = True
                deleted_edges.add((i, j))
                deleted_edges_with_p[(i, j)] = p_ij
                printv("deleted")
                break

            # calculate t_ji (regressing i on j) and find its p-value
            ssr1_ji = np.sum(np.square( X1_i - X1_mj @ b1_mji.T ))
            ssr2_ji = np.sum(np.square( X2_i - X2_mj @ b2_mji.T ))
            var1_ji = ssr1_ji / (n1 - len(m_i))
            var2_ji = ssr2_ji / (n2 - len(m_i))
            b1_ji = b1_mji[-1]
            b2_ji = b2_mji[-1]
            t_ji = (b1_ji - b2_ji)**2 * np.linalg.inv(var1_ji * K1_ji + var2_ji * K2_ji)[-1, -1]
            p_ji = 1 - stats.f.cdf(t_ji, 1, n1 + n2 - len(m_i) - len(m_j))
            if p_ji > alpha_:  # accept hypothesis that b1_ji = b2_ji, delete edge
                is_regression_invariant = True
                deleted_edges.add((i, j))
                deleted_edges_with_p[(i, j)] = p_ji
                printv("deleted")
                break

            max_p = max(max_p, p_ij, p_ji)
        # end of inner loop of powerset

        if not is_regression_invariant:
            printv("retained")
            retained_edges.add((i, j))
            retained_edges_with_p[(i, j)] = max_p

    if isinstance(alpha, Iterable):
        retained_edges_dict = {alpha_: retained_edges}
        deleted_edges_dict = {alpha_: deleted_edges}
        for a in set(alpha) - {alpha_}:
            # if edge was deleted for highest alpha, it would have been deleted for lower alphas.
            deleted_edges_dict[a] = deleted_edges.copy()
            retained_edges_dict[a] = set()
            for (i, j), p in retained_edges_with_p.items():
                if p > a:
                    deleted_edges_dict[a].add((i, j))
                else:
                    retained_edges_dict[a].add((i, j))
        printv("Retained edges: % s" % {k: sorted(r) for k, r in retained_edges_dict.items()})
        return retained_edges_dict, deleted_edges_dict
    else:
        printv("Retained edges: % s" % sorted(retained_edges))
        return retained_edges, retained_edges_with_p, deleted_edges, deleted_edges_with_p


def estimate_dug(K1, K2, n1, n2, alpha=.001, verbose=False):
    p = K1.shape[0]
    K1_zeroed = math_utils.zero_hyp_test(K1, n1, alpha)
    K2_zeroed = math_utils.zero_hyp_test(K2, n2, alpha)

    same_sign = ((K1_zeroed > 0) & (K2_zeroed > 0)) | ((K1_zeroed < 0) & (K2_zeroed < 0))
    same_sign_diff_mag = np.zeros(K1.shape).astype('bool')
    for i, j in math_utils.upper_tri_ixs_nonzero(same_sign):
        t = (K1[i, j] * (n1 - p + 1) - K2[i, j] * (n2 - p + 1))**2
        d = K1[i,i] * K1[j,j] * (n1 - p + 1) + K2[i,i] * K2[j,j] * (n2 - p + 1)
        d += K1[i,j] * K1[i,j] * (n1 - p + 1) + K2[i,j] * K2[i,j] * (n2 - p + 1)
        p = 1 - stats.f.cdf(t / d, 1, n1 + n2 - 2 * p + 2)
        if p < alpha:
            same_sign_diff_mag[i, j] = True
            same_sign_diff_mag[j, i] = True

    different_signs = ((K1_zeroed > 0) & (K2_zeroed < 0)) | ((K1_zeroed < 0) & (K2_zeroed > 0))
    only_first_zero = (K1_zeroed == 0) & (K2_zeroed != 0)
    only_second_zero = (K1_zeroed != 0) & (K2_zeroed == 0)
    K_diff = only_first_zero | only_second_zero | different_signs | same_sign_diff_mag

    diag = infer_diagonals(K1, K2, n1, n2, alpha=alpha)
    estimated_dug = set(math_utils.upper_tri_ixs_nonzero(K_diff))
    changed_nodes = diag | {i for i, j in estimated_dug} | {j for i, j in estimated_dug}
    return estimated_dug, changed_nodes


def infer_diagonals(K1, K2, n1, n2, alpha=.001):
    p = K1.shape[0]
    diag = set()
    for i in range(p):
        sigma1 = K1[i, i] ** -1
        sigma2 = K2[i, i] ** -1
        p = stats.f.cdf(sigma1 / sigma2, n1 - p + 1, n2 - p + 1)
        p = min(p, 1-p)
        if p < alpha:
            diag.add(i)

    return diag
