# -*- encoding: utf-8 -*-

"""
############################
"""

from logging import getLogger

import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

#    TopK Metrics    #


def recall_(pos_index, pos_len):
    # Recall: average single users recall ratio.
    rec_ret = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
    return rec_ret.mean(axis=0)


def ndcg_(pos_index, pos_len):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality.
    Through normalizing the score, users and their recommendation list results in the whole test set can be evaluated.
    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in u^{te}NDCG_u@K}}{|u^{te}|}
        \end{gather}
    :math:`K` stands for recommending :math:`K` items.
    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.
    :math:`2^{rel_i}` equals to 1 if the item hits otherwise 0.
    :math:`U^{te}` is for all users in the test set.
    """
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result.mean(axis=0)


def map_(pos_index, pos_len):
    r"""MAP_ (also known as Mean Average Precision) The MAP is meant to calculate Avg. Precision for the relevant items.
    Note:
        In this case the normalization factor used is :math:`\frac{1}{\min (m,N)}`, which prevents your AP score from
        being unfairly suppressed when your number of recommendations couldn't possibly capture all the correct ones.
    .. _map: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms
    .. math::
        \begin{align*}
        \mathrm{AP@N} &= \frac{1}{\mathrm{min}(m,N)}\sum_{k=1}^N P(k) \cdot rel(k) \\
        \mathrm{MAP@N}& = \frac{1}{|U|}\sum_{u=1}^{|U|}(\mathrm{AP@N})_u
        \end{align*}
    """
    pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    sum_pre = np.cumsum(pre * pos_index.astype(float), axis=1)
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
    result = np.zeros_like(pos_index, dtype=float)
    for row, lens in enumerate(actual_len):
        ranges = np.arange(1, pos_index.shape[1]+1)
        ranges[lens:] = ranges[lens - 1]
        result[row] = sum_pre[row] / ranges
    return result.mean(axis=0)


def precision_(pos_index, pos_len):
    r"""Precision_ (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances
    .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision
    .. math::
        \mathrm {Precision@K} = \frac{|Rel_u \cap Rec_u|}{Rec_u}
    :math:`Rel_u` is the set of items relavent to user :math:`U`,
    :math:`Rec_u` is the top K items recommended to users.
    We obtain the result by calculating the average :math:`Precision@K` of each user.
    """
    rec_ret = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    return rec_ret.mean(axis=0)


def mrr_(pos_index, pos_len):
    r"""The MRR_ (also known as mean reciprocal rank) is a statistic measure for evaluating any process
    that produces a list of possible responses to a sample of queries, ordered by probability of correctness.
    .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    .. math::
        \mathrm {MRR} = \frac{1}{|{U}|} \sum_{i=1}^{|{U}|} \frac{1}{rank_i}
    :math:`U` is the number of users, :math:`rank_i` is the rank of the first item in the recommendation list
    in the test set results for user :math:`i`.
    """
    idxs = pos_index.argmax(axis=1)
    result = np.zeros_like(pos_index, dtype=float)
    for row, idx in enumerate(idxs):
        if pos_index[row, idx] > 0:
            result[row, idx:] = 1 / (idx + 1)
        else:
            result[row, idx:] = 0
    return result.mean(axis=0)


#    CTR Metrics    #

def auc_(trues, preds):
    r"""AUC_ (also known as Area Under Curve) is used to evaluate the two-class model, referring to
    the area under the ROC curve

    .. _AUC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

    Note:
        This metric does not calculate group-based AUC which considers the AUC scores
        averaged across users. It is also not limited to k. Instead, it calculates the
        scores on the entire prediction results regardless the users.

    .. math::
        \mathrm {AUC} = \frac{\sum\limits_{i=1}^M rank_{i}
        - {{M} \times {(M+1)}}} {{M} \times {N}}

    :math:`M` is the number of positive samples.
    :math:`N` is the number of negative samples.
    :math:`rank_i` is the rank of the ith positive sample.

    """
    fps, tps = _binary_clf_curve(trues, preds)

    if len(fps) > 2:
        optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0:
        logger = getLogger()
        logger.warning("No negative samples in y_true, "
                       "false positive value should be meaningless")
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        logger = getLogger()
        logger.warning("No positive samples in y_true, "
                       "true positive value should be meaningless")
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return sk_auc(fpr, tpr)


# Loss based Metrics #

def mae_(trues, preds):
    r"""`Mean absolute error regression loss`__

    .. __: https://en.wikipedia.org/wiki/Mean_absolute_error

    .. math::
        \mathrm{MAE}=\frac{1}{|{T}|} \sum_{(u, i) \in {T}}\left|\hat{r}_{u i}-r_{u i}\right|

    :math:`T` is the test set, :math:`\hat{r}_{u i}` is the score predicted by the model,
    and :math:`r_{u i}` the actual score of the test set.

    """
    return mean_absolute_error(trues, preds)


def rmse_(trues, preds):
    r"""`Mean std error regression loss`__

    .. __: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    .. math::
        \mathrm{RMSE} = \sqrt{\frac{1}{|{T}|} \sum_{(u, i) \in {T}}(\hat{r}_{u i}-r_{u i})^{2}}

    :math:`T` is the test set, :math:`\hat{r}_{u i}` is the score predicted by the model,
    and :math:`r_{u i}` the actual score of the test set.

    """
    return np.sqrt(mean_squared_error(trues, preds))


def log_loss_(trues, preds):
    r"""`Log loss`__, aka logistic loss or cross-entropy loss

    .. __: http://wiki.fast.ai/index.php/Log_Loss

    .. math::
        -\log {P(y_t|y_p)} = -(({y_t}\ \log{y_p}) + {(1-y_t)}\ \log{(1 - y_p)})

    For a single sample, :math:`y_t` is true label in :math:`\{0,1\}`.
    :math:`y_p` is the estimated probability that :math:`y_t = 1`.

    """
    eps = 1e-15
    preds = np.float64(preds)
    preds = np.clip(preds, eps, 1 - eps)
    loss = np.sum(- trues * np.log(preds) - (1 - trues) * np.log(1 - preds))

    return loss / len(preds)


def _binary_clf_curve(trues, preds):
    """Calculate true and false positives per binary classification threshold

    Args:
        trues (numpy.ndarray): the true scores' list
        preds (numpy.ndarray): the predict scores' list

    Returns:
        fps (np.ndarray): A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]
        preds (numpy.ndarray): An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i].

    Note:
        To improve efficiency, we referred to the source code(which is available at sklearn.metrics.roc_curve)
        in SkLearn and made some optimizations.

    """
    trues = (trues == 1)

    desc_idxs = np.argsort(preds)[::-1]
    preds = preds[desc_idxs]
    trues = trues[desc_idxs]

    unique_val_idxs = np.where(np.diff(preds))[0]
    threshold_idxs = np.r_[unique_val_idxs, trues.size - 1]

    tps = np.cumsum(trues)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps


# Item based Metrics #

# TODO
# def coverage_():
#     raise NotImplementedError


# def gini_index_():
#     raise NotImplementedError


# def shannon_entropy_():
#     raise NotImplementedError


# def diversity_():
#     raise NotImplementedError


"""Function name and function mapper.
Useful when we have to serialize evaluation metric names
and call the functions based on deserialized names
"""
metrics_dict = {
    'ndcg': ndcg_,
    'precision': precision_,
    'map': map_,
    'recall': recall_,
    'mrr': mrr_,
    'rmse': rmse_,
    'mae': mae_,
    'logloss': log_loss_,
    'auc': auc_
}
