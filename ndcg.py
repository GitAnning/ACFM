import numpy as np

# 计算DCG
def dcg(pred, k):
    relevance_scores = np.asarray(pred)[:k]
    if relevance_scores.size == 0:
        return 0.0
    return np.sum((2**pred - 1) / np.log2(np.arange(1, pred.size + 1) + 1))

# 计算IDCG
def idcg(label, k):
    sorted_relevance_scores = label
    return dcg(sorted_relevance_scores, k)

# 计算NDCG
def ndcg(label, pred):
    k = len(pred)
    dcg_score = dcg(pred, k)
    idcg_score = idcg(label, k)
    if idcg == 0:
        return 0.0
    return dcg_score / idcg_score

