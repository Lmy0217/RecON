import time

import numpy as np
import torch
import torch.nn.functional as F

import configs
import datasets
import utils

__all__ = ['Spine']


class Spine(datasets.BaseDataset):

    @staticmethod
    def more(cfg):
        cfg.source.elements = cfg.source.width * cfg.source.height * cfg.source.channel
        cfg.paths.source = configs.env.getdir(cfg.paths.source)
        cfg.paths.target = configs.env.getdir(cfg.paths.target)

        cfg.num_workers = 0
        cfg.pin_memory = False

        cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        return cfg

    def load(self):
        source_raw = np.load(self.cfg.paths.source, allow_pickle=True)[()]
        target_raw = np.load(self.cfg.paths.target, allow_pickle=True)[()]

        source, target_dof, target_point = [], [], []
        for k1 in sorted(source_raw.keys()):
            for k2 in sorted(source_raw[k1].keys()):
                for k3 in sorted(source_raw[k1][k2].keys()):
                    slices = torch.from_numpy(source_raw[k1][k2][k3])
                    source.append(slices)
                    tp = torch.from_numpy(target_raw[k1][k2][k3]['point'].astype(np.float32))
                    tp = tp.view(-1, 3, 3)
                    tp = torch.cat([
                        tp[:, 0:1, :],
                        tp[:, 0:1, :] + (tp[:, 1:2, :] * self.cfg.source.height / 2 - tp[:, 2:3, :] * self.cfg.source.width / 2),
                        tp[:, 0:1, :] + (tp[:, 1:2, :] * self.cfg.source.height / 2 + tp[:, 2:3, :] * self.cfg.source.width / 2)
                    ], dim=1)
                    tp = tp.view(-1, 9)
                    tp, td = self.preprocessing(tp)
                    target_point.append(tp)
                    target_dof.append(td)

        trainset_length = int(self.cfg.series_per_data[0] * self.cfg.train_test_range[0])
        valset_length = int(self.cfg.series_per_data[1] * self.cfg.train_test_range[1])
        testset_length = int(self.cfg.series_per_data[2] * self.cfg.train_test_range[2])
        data_count = trainset_length + valset_length + testset_length

        return {'source': source, 'target_dof': target_dof, 'target_point': target_point}, data_count

    def preprocessing(self, tp):
        tp = tp.view(-1, 3, 3)
        pall = torch.cat([tp, 2 * tp[:, 0:1, :] - tp[:, 1:2, :], 2 * tp[:, 0:1, :] - tp[:, 2:3, :]], dim=1)
        min_loca = torch.min(pall.reshape(-1, 3), dim=0)[0]
        tp = tp - min_loca.unsqueeze(0).unsqueeze(0)
        td = utils.simulation.series_to_dof(tp)
        tp = tp.view(-1, 9)
        return tp, td

    def __getitem__(self, index):
        idx = self.get_idx(index)

        source = self.data['source'][idx].to(self.cfg.device)
        target_point = self.data['target_point'][idx]

        frame_rate = torch.randint(self.cfg.frame_rate[0], self.cfg.frame_rate[1] + 1, (1,))
        source = source[::frame_rate]
        target_point = target_point[::frame_rate]
        target_point, target_dof = self.preprocessing(target_point.view(-1, 3, 3))
        optical_flow = utils.image.get_optical_flow(source, device=self.cfg.device)
        edge = utils.image.get_edge(source, device=self.cfg.device)

        source_out = source.unsqueeze(1)
        target_out = torch.cat([F.pad(target_dof, (0, 0, 0, 1)), target_point.view(-1, 9)], dim=-1)

        sample_dict = {
            'source': source_out, 'target': target_out,
            'optical_flow': optical_flow,
            'edge': edge,
            'frame_rate': frame_rate,
            'info': torch.tensor(len(source_out))
        }

        utils.common.set_seed(int(time.time() * 1000) % (1 << 32) + index)
        return sample_dict, index
