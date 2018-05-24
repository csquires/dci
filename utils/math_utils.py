import itertools as itr
from scipy import stats
import numpy as np
import math


# === RANDOM NUMBERS
def bernoulli(p):
    return np.random.binomial(1, p)


def rademacher():
    return (np.random.binomial(1, .5) - .5) * 2


def uniform_in(bounds, size=None):
    lens = np.array([bound[1] - bound[0] for bound in bounds])
    probs = lens/lens.sum()
    if size is None:
        bound_ix = np.random.choice(len(bounds), p=probs)
        return np.random.uniform(bounds[bound_ix][0], bounds[bound_ix][1])
    else:
        samples = np.zeros(size)
        for ix in np.ndindex(samples.shape):
            bound_ix = np.random.choice(len(bounds), p=probs)
            samples[ix] = np.random.uniform(bounds[bound_ix][0], bounds[bound_ix][1])
        return samples


# === HYPOTHESIS TESTING
def ftest_pvalue(var1, var2, d1, d2):
    """
    Probability that empirical variances var1 and var2 would arise from same true variance

    :param var1: first empirical variance
    :param var2: second empirical variance
    :param d1: degrees of freedom for var1
    :param d2: degrees of freedom for var2
    :return:
    """
    f = min(var1, var2) / max(var1, var2)
    d1_, d2_ = (d1, d2) if var1 < var2 else (d2, d1)
    return 2 * stats.f.cdf(f, d1_, d2_)


def zero_hyp_test(K, n, alpha):
    p = K.shape[0]
    K_zeroed = K.copy()
    for i, j in itr.combinations(range(p), 2):
        rho = - K[i, j] / np.sqrt(K[i, i] * K[j, j])
        z = math.atanh(rho)
        p = np.sqrt(n - p - 1) * abs(z)
        if p < stats.norm.ppf(1 - alpha/2.):
            K_zeroed[i, j] = 0.
            K_zeroed[j, i] = 0.
    return K_zeroed


def powerset(s, max_set_size=None):
    """
    Return all subsets up to max_set_size, in order from smallest to largest subsets.
    By default, return all subsets.

    :param s: set
    :param max_set_size: maximum subset size
    :return:
    """
    if max_set_size is None:
        max_set_size = len(s)
    return itr.chain.from_iterable(itr.combinations(s, r) for r in range(max_set_size+1))


# === DAG INDEXING
def upper_tri_ixs(p):
    """
    Return the indices corresponding to the upper triangle of a (p x p) matrix
    :param p:
    :return:
    """
    return itr.combinations(range(p), 2)


def upper_tri_ixs_nonzero(mat, thresh=0):
    """
    Return iterable of (i, j) such that mat[i,j] > thresh (default 0) and i < j
    :param mat:
    :param thresh:
    :return:
    """
    return filter(lambda ix: abs(mat[ix]) > thresh, upper_tri_ixs(mat.shape[0]))


def upper_tri_ixs_zero(mat):
    """
    Return iterable of (i, j) such that mat[i,j] > thresh (default 0) and i < j
    :param mat:
    :return:
    """
    return filter(lambda ix: abs(mat[ix]) == 0, upper_tri_ixs(mat.shape[0]))


# === RANDOM DAGs
def random_dag(p, sparsity, lower_bound=.25, upper_bound=1.):
    B = np.zeros([p, p])
    for i, j in upper_tri_ixs(p):
        exists = bernoulli(sparsity)
        magnitude = np.random.uniform(lower_bound, upper_bound)
        sign = rademacher()
        B[i, j] = exists * magnitude * sign
    return B


def random_dag_changes(mat, r=0, a=0, c=0):
    nonzero_ixs = set(zip(*np.where(mat != 0)))
    zero_ixs = upper_tri_ixs_zero(mat)
    new_mat, removed_ixs = random_removal(mat, r, indices=nonzero_ixs)
    remaining_nonzero_ixs = nonzero_ixs - removed_ixs
    new_mat, added_ixs = random_addition(new_mat, a, indices=zero_ixs)
    new_mat, changed_ixs = random_change(new_mat, c, indices=remaining_nonzero_ixs)

    changed_edges = removed_ixs | added_ixs | changed_ixs
    return new_mat, changed_edges


def random_removal(B, removal_prob, indices=None):
    B_ = B.copy()
    if indices is None:
        indices = zip(*np.where(B != 0))

    removed_indices = set()
    for i, j in indices:
        if bernoulli(removal_prob) == 1:
            removed_indices.add((i, j))
            B_[i, j] = 0

    return B_, removed_indices


def random_addition(B, a, indices=None, lower_bound=.25, upper_bound=1.):
    B_ = B.copy()
    if indices is None:
        indices = upper_tri_ixs_zero(B)

    added_indices = set()
    for i, j in indices:
        if bernoulli(a):
            added_indices.add((i, j))
            magnitude = np.random.uniform(lower_bound, upper_bound)
            sign = rademacher()
            B_[i, j] = magnitude * sign

    return B_, added_indices


def random_change(B, change_prob, indices=None, lower_bound=.25, upper_bound=1.):
    B_ = B.copy()
    if indices is None:
        indices = zip(*np.where(B != 0))

    changed_indices = set()
    for i, j in indices:
        if bernoulli(change_prob):
            changed_indices.add((i, j))
            val = B[i, j]
            s = np.sign(val)
            opp_sign_range = sorted([-s*lower_bound, -s*upper_bound])
            lower_range_ub = max(abs(val)-lower_bound, lower_bound)
            higher_range_lb = min(abs(val)+lower_bound, upper_bound)
            same_sign_lower_range = sorted([s * lower_bound, s * lower_range_ub])
            same_sign_upper_range = sorted([s * higher_range_lb, s * upper_bound])
            all_bounds = [opp_sign_range, same_sign_lower_range, same_sign_upper_range]
            B_[i, j] = uniform_in(all_bounds)

    return B_, changed_indices


def sample_dag(B, n, sigmas=None):
    p = B.shape[0]
    if sigmas is None:
        sigmas = np.ones(p)
    noise = np.random.normal(scale=sigmas, size=(n, p))
    A = np.linalg.inv(np.eye(p) - B.T)
    return noise @ A.T


def adj2prec(B, sigmas=None):
    p = B.shape[0]
    if sigmas is None:
        sigmas = np.ones(p)
    I = np.eye(p)
    return (I - B) @ np.diag(sigmas) @ (I - B).T


# === ANALYSIS
def sort_pos_neg(estimated, actual_positives, actual_negatives):
    true_pos = actual_positives & estimated
    true_neg = actual_negatives - estimated
    false_pos = actual_negatives & estimated
    false_neg = actual_positives - estimated
    return true_pos, true_neg, false_pos, false_neg


def compute_pos_neg_rates(true_pos, true_neg, false_pos, false_neg):
    total_pos = len(true_pos) + len(false_neg)
    total_neg = len(true_neg) + len(false_pos)

    tpr = len(true_pos) / total_pos if total_pos != 0 else 1
    tnr = len(true_neg) / total_neg if total_neg != 0 else 1
    fpr = len(false_pos) / total_neg if total_neg != 0 else 0
    fnr = len(false_neg) / total_pos if total_pos != 0 else 0

    return tpr, tnr, fpr, fnr
