# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   lbr -> evaluate
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   06/04/2022 11:56
=================================================='''
import numpy as np


def ap_k(query_labels, k=1):
    ap = 0.0
    positives = 0
    for idx, i in enumerate(query_labels[:k]):
        if i == 1:
            positives += 1
            ap_at_count = positives / (idx + 1)
            ap += ap_at_count
    return ap / k


def recall_k(query_labels, k=1):
    return np.sum(query_labels) / k


def eval_k(preds, gts, k=1):
    output = {}
    mean_ap = []
    mean_recall = []
    for query in preds.keys():
        pred_cands = preds[query]
        gt_cands = gts[query]
        pred = []

        for c in pred_cands[:k]:
            if c in gt_cands:
                pred.append(1)
            else:
                pred.append(0)
        ap = ap_k(query_labels=pred, k=k)
        recall = recall_k(query_labels=pred, k=len(gt_cands))
        mean_ap.append(ap)
        mean_recall.append(recall)

        # print("{:s} topk: {:d} ap: {:.4f} recall: {:.4f}".format(query, k, ap, recall))

        output[query] = (ap, recall)

    return np.mean(mean_ap), np.mean(mean_recall), output


def evaluate_retrieval(preds, gts, ks=[1, 10, 20, 50]):
    output = {}
    for k in ks:
        mean_ap, mean_recall, _ = eval_k(preds=preds, gts=gts, k=k)
        output[k] = {
            'accuracy': mean_ap,
            'recall': mean_recall,
        }
    return output


def evaluate_retrieval_by_query(preds, gts, ks=[1, 10, 20, 50]):
    output = {}
    for k in ks:
        output[k] = 0

    failed_cases = []
    for q in preds.keys():
        gt_cans = gts[q]
        for k in ks:
            pred_cans = preds[q][:k]
            overlap = [v for v in pred_cans if v in gt_cans]
            if len(overlap) >= 1:
                output[k] += 1

            if k == 50 and len(overlap) == 0:
                failed_cases.append(q)

    for k in ks:
        output[k] = output[k] / len(preds.keys())

    output['failed_case'] = failed_cases
    return output
