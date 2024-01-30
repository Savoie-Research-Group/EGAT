from typing import Callable

import torch
import torch.nn as nn
import numpy as np


def mcc_class_loss(
    predictions: torch.tensor,
    targets: torch.tensor,
    data_weights: torch.tensor,
    mask: torch.tensor,
) -> torch.tensor:
    """
    A classification loss using a soft version of the Matthews Correlation Coefficient.

    :param predictions: Model predictions with shape(batch_size, tasks).
    :param targets: Target values with shape(batch_size, tasks).
    :param data_weights: A tensor with float values indicating how heavily to weight each datapoint in training with shape(batch_size, 1)
    :param mask: A tensor with boolean values indicating whether the loss for this prediction is considered in the gradient descent with shape(batch_size, tasks).
    :return: A tensor containing loss values of shape(tasks).
    """
    # shape(batch, tasks)
    # (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    TP = torch.sum(targets * predictions * data_weights * mask, axis=0)
    FP = torch.sum((1 - targets) * predictions * data_weights * mask, axis=0)
    FN = torch.sum(targets * (1 - predictions) * data_weights * mask, axis=0)
    TN = torch.sum((1 - targets) * (1 - predictions) * data_weights * mask, axis=0)
    loss = 1 - ((TP * TN - FP * FN) / torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    return loss


def mcc_multiclass_loss(
    predictions: torch.tensor,
    targets: torch.tensor,
    data_weights: torch.tensor,
    mask: torch.tensor,
) -> torch.tensor:
    """
    A multiclass loss using a soft version of the Matthews Correlation Coefficient. Multiclass definition follows the version in sklearn documentation (https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-correlation-coefficient).

    :param predictions: Model predictions with shape(batch_size, classes).
    :param targets: Target values with shape(batch_size).
    :param data_weights: A tensor with float values indicating how heavily to weight each datapoint in training with shape(batch_size, 1)
    :param mask: A tensor with boolean values indicating whether the loss for this prediction is considered in the gradient descent with shape(batch_size).
    :return: A tensor value for the loss.
    """
    torch_device = predictions.device
    mask = mask.unsqueeze(1)

    bin_targets = torch.zeros_like(predictions, device=torch_device)
    bin_targets[torch.arange(predictions.shape[0]), targets] = 1

    pred_classes = predictions.argmax(dim=1)
    bin_preds = torch.zeros_like(predictions, device=torch_device)
    bin_preds[torch.arange(predictions.shape[0]), pred_classes] = 1

    masked_data_weights = data_weights * mask

    t_sum = torch.sum(bin_targets * masked_data_weights, axis=0)  # number of times each class truly occurred
    p_sum = torch.sum(bin_preds * masked_data_weights, axis=0)  # number of times each class was predicted

    n_correct = torch.sum(bin_preds * bin_targets * masked_data_weights)  # total number of samples correctly predicted
    n_samples = torch.sum(predictions * masked_data_weights)  # total number of samples

    cov_ytyp = n_correct * n_samples - torch.dot(p_sum, t_sum)
    cov_ypyp = n_samples**2 - torch.dot(p_sum, p_sum)
    cov_ytyt = n_samples**2 - torch.dot(t_sum, t_sum)

    if cov_ypyp * cov_ytyt == 0:
        loss = torch.tensor(1.0, device=torch_device)
    else:
        mcc = cov_ytyp / torch.sqrt(cov_ytyt * cov_ypyp)
        loss = 1 - mcc

    return loss


# evidential classification
def dirichlet_class_loss(alphas, target_labels, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al in classification datasets.
    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, tasks*2).
                   Negative class first then positive class in dimension 1.
    :param target_labels: Digital labels to predict in shape(datapoints, tasks).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    torch_device = alphas.device
    num_tasks = target_labels.shape[1]
    num_classes = 2
    alphas = torch.reshape(alphas, (alphas.shape[0], num_tasks, num_classes))

    y_one_hot = torch.eye(num_classes, device=torch_device)[target_labels.long()]

    return dirichlet_common_loss(alphas=alphas, y_one_hot=y_one_hot, lam=lam)


def dirichlet_multiclass_loss(alphas, target_labels, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al for multiclass datasets.
    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, task, classes).
    :param target_labels: Digital labels to predict in shape(datapoints, tasks).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    torch_device = alphas.device
    num_classes = alphas.shape[2]

    y_one_hot = torch.eye(num_classes, device=torch_device)[target_labels.long()]

    return dirichlet_common_loss(alphas=alphas, y_one_hot=y_one_hot, lam=lam)


def dirichlet_common_loss(alphas, y_one_hot, lam=0):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al. This function follows
    after the classification and multiclass specific functions that reshape the
    alpha inputs and create one-hot targets.

    :param alphas: Predicted parameters for Dirichlet in shape(datapoints, task, classes).
    :param y_one_hot: Digital labels to predict in shape(datapoints, tasks, classes).
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    # SOS term
    S = torch.sum(alphas, dim=-1, keepdim=True)
    p = alphas / S
    A = torch.sum((y_one_hot - p) ** 2, dim=-1, keepdim=True)
    B = torch.sum((p * (1 - p)) / (S + 1), dim=-1, keepdim=True)
    SOS = A + B

    alpha_hat = y_one_hot + (1 - y_one_hot) * alphas

    beta = torch.ones_like(alpha_hat)
    S_alpha = torch.sum(alpha_hat, dim=-1, keepdim=True)
    S_beta = torch.sum(beta, dim=-1, keepdim=True)

    ln_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha_hat), dim=-1, keepdim=True)
    ln_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)

    # digamma terms
    dg_alpha = torch.digamma(alpha_hat)
    dg_S_alpha = torch.digamma(S_alpha)

    # KL
    KL = ln_alpha + ln_beta + torch.sum((alpha_hat - beta) * (dg_alpha - dg_S_alpha), dim=-1, keepdim=True)

    KL = lam * KL

    # loss = torch.mean(SOS + KL)
    loss = SOS + KL
    loss = torch.mean(loss, dim=-1)
    return loss

