import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def average_drift_rate(predict, target, eps=0.1):
    drift = torch.norm(target[1:] - predict[1:], 2, dim=-1)
    dist = torch.cumsum(torch.norm(target[1:] - target[:-1], 2, dim=-1), dim=0)
    if eps is not None:
        flag = dist >= eps
        drift = drift[flag]
        dist = dist[flag]
    return torch.mean(drift / dist, dim=0)


def get_adr(real_series, fake_series):
    adr_pc = average_drift_rate(fake_series[:, 0, :], real_series[:, 0, :])
    adr_p1 = average_drift_rate(fake_series[:, 1, :], real_series[:, 1, :])
    adr_p2 = average_drift_rate(fake_series[:, 2, :], real_series[:, 2, :])
    return (adr_pc + adr_p1 + adr_p2) / 3


if __name__ == '__main__':
    device = torch.device("cuda:0")
    dir_save = r'../save/online_fm-hp_fm-Spine/RecON'

    MEA, FDR, ADR, MD, SD, HD = [], [], [], [], [], []
    for file in tqdm(sorted(os.listdir(dir_save))):
        value = torch.load(os.path.join(dir_save, file), map_location=device)
        mea, fdr, adr, md, sd, hd = [], [], [], [], [], []
        for idx, loss in enumerate(value['loss']):
            mea.append(loss['MEA'])
            fdr.append(loss['FDR'])
            a = get_adr(value['real_series'], value['fake_series'][idx])
            adr.append(a)
            md.append(loss['MD'])
            sd.append(loss['SD'])
            hd.append(loss['HD'])
        MEA.append(torch.tensor(mea, device=device))
        FDR.append(torch.tensor(fdr, device=device))
        ADR.append(torch.tensor(adr, device=device))
        MD.append(torch.tensor(md, device=device))
        SD.append(torch.tensor(sd, device=device))
        HD.append(torch.tensor(hd, device=device))
    MEA = torch.stack(MEA, dim=0)
    FDR = torch.stack(FDR, dim=0)
    ADR = torch.stack(ADR, dim=0)
    MD = torch.stack(MD, dim=0)
    SD = torch.stack(SD, dim=0)
    HD = torch.stack(HD, dim=0)

    MEA = torch.mean(MEA, dim=0).cpu().numpy()
    FDR = torch.mean(FDR, dim=0).cpu().numpy()
    ADR = torch.mean(ADR, dim=0).cpu().numpy()
    MD = torch.mean(MD, dim=0).cpu().numpy()
    SD = torch.mean(SD, dim=0).cpu().numpy()
    HD = torch.mean(HD, dim=0).cpu().numpy()
    x = np.linspace(0, len(MEA) - 1, len(MEA))

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.plot(x, MEA)
    plt.subplot(2, 3, 2)
    plt.plot(x, FDR)
    plt.subplot(2, 3, 3)
    plt.plot(x, ADR)
    plt.subplot(2, 3, 4)
    plt.plot(x, MD)
    plt.subplot(2, 3, 5)
    plt.plot(x, SD)
    plt.subplot(2, 3, 6)
    plt.plot(x, HD)
    plt.show()
