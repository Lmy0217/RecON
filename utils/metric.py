import numpy as np
import scipy.spatial
import torch

import utils


def correlation_loss(labels, outputs, eps=1e-6):
    x = outputs.flatten()
    y = labels.flatten()
    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y

    var_x = torch.sum((x - mean_x) ** 2 / x.shape[0]) + eps
    var_y = torch.sum((y - mean_y) ** 2 / y.shape[0]) + eps

    corr_xy = cov_xy / (torch.sqrt(var_x * var_y))

    loss = 1 - corr_xy
    return loss


def final_drift_rate(predict, target, eps=1e-6):
    final_drift = torch.norm(target[-1] - predict[-1], 2, dim=-1)
    dist = torch.sum(torch.norm(target[1:] - target[:-1], 2, dim=-1)) + eps
    return final_drift / dist


def average_drift_rate(predict, target, eps=0.1):
    drift = torch.norm(target[1:] - predict[1:], 2, dim=-1)
    dist = torch.cumsum(torch.norm(target[1:] - target[:-1], 2, dim=-1), dim=0)
    if eps is not None:
        flag = dist >= eps
        drift = drift[flag]
        dist = dist[flag]
    return torch.mean(drift / dist, dim=0)


def max_drift(predict, target):
    drift = torch.norm(target[1:] - predict[1:], 2, dim=-1)
    return torch.max(drift, dim=0)[0]


def sum_drift(predict, target):
    drift = torch.norm(target[1:] - predict[1:], 2, dim=-1)
    return torch.sum(drift)


def symmetric_hausdorff_distance(predict, target):
    h_pt = scipy.spatial.distance.directed_hausdorff(predict, target)[0]
    h_tp = scipy.spatial.distance.directed_hausdorff(target, predict)[0]
    return max(h_tp, h_pt)


def get_metric(real_series, fake_series):
    metric_dict = {}

    real_axis = utils.simulation.get_axis(real_series)
    fake_axis = utils.simulation.get_axis(fake_series)
    cos = torch.sum(real_axis * fake_axis, dim=-1)
    cos.clamp_(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    angle = torch.acos(cos) * 180 / np.pi
    metric_dict['MEA'] = torch.mean(angle)

    fdr_pc = final_drift_rate(fake_series[:, 0, :], real_series[:, 0, :])
    fdr_p1 = final_drift_rate(fake_series[:, 1, :], real_series[:, 1, :])
    fdr_p2 = final_drift_rate(fake_series[:, 2, :], real_series[:, 2, :])
    metric_dict['FDR'] = (fdr_pc + fdr_p1 + fdr_p2) / 3

    adr_pc = average_drift_rate(fake_series[:, 0, :], real_series[:, 0, :])
    adr_p1 = average_drift_rate(fake_series[:, 1, :], real_series[:, 1, :])
    adr_p2 = average_drift_rate(fake_series[:, 2, :], real_series[:, 2, :])
    metric_dict['ADR'] = (adr_pc + adr_p1 + adr_p2) / 3

    md_pc = max_drift(fake_series[:, 0, :], real_series[:, 0, :])
    md_p1 = max_drift(fake_series[:, 1, :], real_series[:, 1, :])
    md_p2 = max_drift(fake_series[:, 2, :], real_series[:, 2, :])
    metric_dict['MD'] = (md_pc + md_p1 + md_p2) / 3

    sd_pc = sum_drift(fake_series[:, 0, :], real_series[:, 0, :])
    sd_p1 = sum_drift(fake_series[:, 1, :], real_series[:, 1, :])
    sd_p2 = sum_drift(fake_series[:, 2, :], real_series[:, 2, :])
    metric_dict['SD'] = (sd_pc + sd_p1 + sd_p2) / 3

    fake_series_cpu, real_series_cpu = fake_series.cpu().numpy(), real_series.cpu().numpy()
    hausdorff_pc = symmetric_hausdorff_distance(fake_series_cpu[:, 0, :], real_series_cpu[:, 0, :])
    hausdorff_p1 = symmetric_hausdorff_distance(fake_series_cpu[:, 1, :], real_series_cpu[:, 1, :])
    hausdorff_p2 = symmetric_hausdorff_distance(fake_series_cpu[:, 2, :], real_series_cpu[:, 2, :])
    metric_dict['HD'] = (hausdorff_pc + hausdorff_p1 + hausdorff_p2) / 3

    metric_dict = {k: v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=real_series.dtype, device=real_series.device) for k, v in metric_dict.items()}

    return metric_dict
