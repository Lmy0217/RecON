import warnings

import cv2
import numpy as np
import torch

from models.layers.canny import Canny2D


def get_optical_flow(slices, device):
    height, width = slices.shape[-2], slices.shape[-1]
    if not hasattr(get_optical_flow, 'of'):
        if hasattr(cv2, 'cuda_NvidiaOpticalFlow_2_0'):
            of = cv2.cuda_NvidiaOpticalFlow_2_0.create((width, height), 5, 1, 1, False, False, False, device.index)
        elif hasattr(cv2, 'cuda_NvidiaOpticalFlow_1_0'):
            of = cv2.cuda_NvidiaOpticalFlow_1_0.create(width, height, 5, False, False, False, device.index)
            warnings.warn('use cuda_NvidiaOpticalFlow_1_0!')
        else:
            warnings.warn('opencv-python not support cuda!')
            return None
        get_optical_flow.of = of

    of = get_optical_flow.of
    slices_np = slices.type(torch.uint8).cpu().numpy()
    flows = []
    for i in range(1, len(slices)):
        flow = of.calc(slices_np[i - 1, :, :], slices_np[i, :, :], None)
        if hasattr(cv2, 'cuda_NvidiaOpticalFlow_2_0'):
            flow = of.convertToFloat(flow[0], None)
        else:
            flow = of.upSampler(flow[0], width, height, of.getGridSize(), None)
        flows.append(flow)
    flows = np.stack(flows, axis=0).transpose((0, 3, 1, 2))
    flows = torch.from_numpy(flows).type(slices.dtype).to(slices.device)
    return flows


def get_edge(slices, device, threshold=0.1):
    if not hasattr(get_edge, 'canny'):
        get_edge.canny = Canny2D(threshold=threshold).to(device)
    with torch.no_grad():
        edge = get_edge.canny(slices.unsqueeze(1))
    return edge
