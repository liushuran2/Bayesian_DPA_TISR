# By Shuran Liu, 2023
import numpy as np
import torch

def interval_confidence(mu_list, b_list, epsilon, num_dropout_ensembles=1):
    mu = np.mean(mu_list,axis=0)
    mu_stack = np.tile(mu, (num_dropout_ensembles, 1, 1))
    mu_stack = np.stack(mu_stack, axis=0)
    left = laplace_cdf(mu_stack - epsilon, mu_list, b_list)
    right = laplace_cdf(mu_stack + epsilon, mu_list, b_list)
    return right - left

def laplace_cdf(x, mu_list, b_list):
    """
    Calculate the CDF of a Laplace distribution.

    Parameters:
        x (float or array-like): The value(s) at which to calculate the CDF.
        mu (float): The mean parameter of the Laplace distribution.
        b (float): The scale parameter of the Laplace distribution.

    Returns:
        The cumulative distribution function of the Laplace distribution evaluated at `x`.
    """
    cdf = 0.5 * np.sign(x - mu_list) * (1 - np.exp(-np.abs(x - mu_list) / b_list))
    return np.mean(cdf, axis=0)

def reliability_diagram(y_true, mu_list, b_list, epsilon=0.04, n_bins=50, threhold=0.05):
    y_true = y_true.squeeze(0).squeeze(0)
    confidence = interval_confidence(mu_list, b_list, epsilon)
    y_pred = np.mean(mu_list, axis=0)
    accuracy = np.logical_and(y_pred>y_true-epsilon, y_pred<y_true+epsilon)
    accuracy = accuracy.astype(int)

    confidence = confidence.reshape(-1)
    accuracy = accuracy.reshape(-1)
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    #threhold
    mask = (y_pred > threhold)
    confidence = confidence[mask]
    accuracy = accuracy[mask]

    bin_corrects = [0] * n_bins
    bin_totals = [0] * n_bins
    bin_confidences = [0] * n_bins

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence, bin_boundaries[1:-1])
    for i in range(n_bins):
        bin_indices_i = np.where(bin_indices == i + 1)[0]
        if len(bin_indices_i) > 0:
            bin_corrects[i] = np.sum(accuracy[bin_indices_i])
            bin_totals[i] = len(bin_indices_i)
            bin_confidences[i] = np.sum(confidence[bin_indices_i])
    return bin_confidences, bin_corrects, bin_totals

def diag_torch(n_bins, confidence, accuracy):
    bin_corrects = torch.zeros((n_bins)).float().cuda()
    bin_confidences = torch.zeros((n_bins)).float().cuda()
    bin_total = torch.zeros((n_bins)).float().cuda()

    bin_boundaries = torch.linspace(torch.min(confidence).data.cpu(), torch.max(confidence).data.cpu(), n_bins + 1).float().cuda()
    bin_indices = torch.bucketize(confidence, bin_boundaries[1:-1])
    for i in range(n_bins):
        bin_indices_i = torch.where(bin_indices == i + 1)[0]
        if len(bin_indices_i) > 0:
            bin_corrects[i] = torch.sum(accuracy[bin_indices_i])
            bin_confidences[i] = torch.sum(confidence[bin_indices_i])
            bin_total[i] = len(bin_indices_i)
    non_zero = bin_total.nonzero()
    bin_confidences = bin_confidences[non_zero] / bin_total[non_zero]
    bin_corrects = bin_corrects[non_zero] / bin_total[non_zero]
    bin_ratio = bin_total[non_zero] / torch.sum(bin_total[non_zero])
    ECE = torch.dot(torch.abs(torch.squeeze(bin_confidences) - torch.squeeze(bin_corrects)), torch.squeeze(bin_ratio))

    return ECE