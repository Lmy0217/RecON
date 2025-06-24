import os
import time
import warnings

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import configs
import datasets
import utils

__all__ = ['DDH']


class DDH(datasets.BaseDataset):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.origin = torch.tensor(self.cfg.source.origin, dtype=torch.float32, device=self.cfg.device)
        self.rad = torch.tensor([self.cfg.source.max_distance], dtype=torch.float32, device=self.cfg.device)
        self.direct_down = '-axis_y'

        self.pd = {}
        self.pr = {}
        self.pi = {}

    @staticmethod
    def more(cfg):
        cfg.source.elements = cfg.source.width * cfg.source.height * cfg.source.channel
        cfg.paths.source = configs.env.getdir(cfg.paths.source)
        cfg.paths.order = configs.env.getdir(cfg.paths.order)

        cfg.num_workers = 0
        cfg.pin_memory = False

        cfg.cuda_data_count = 0
        cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg.load_mode = getattr(cfg, 'load_mode', 'memory')

        cfg.while_max_time = 20

        return cfg

    def load(self):
        if self.cfg.load_mode == 'disk':
            source_data = []
            for file in sorted(os.listdir(self.cfg.paths.source)):
                if file.endswith('.nii.gz'):
                    source = os.path.join(self.cfg.paths.source, file)
                    source_data.append(source)
        else:
            source_data_npy_path = os.path.join(self.cfg.paths.source, self.__class__.__name__ + '_data.npy')
            if not os.path.exists(source_data_npy_path):
                source_data = []
                for file in sorted(os.listdir(self.cfg.paths.source)):
                    if file.endswith('.nii.gz'):
                        source = nib.load(os.path.join(self.cfg.paths.source, file)).get_fdata(dtype=np.float32)
                        source_data.append(source)
                np.save(source_data_npy_path, source_data)
            else:
                source_data = np.load(source_data_npy_path, allow_pickle=True)

        if self.cfg.load_mode == 'memory':
            source_data = [torch.from_numpy(s).to(self.cfg.device) if i < self.cfg.cuda_data_count else torch.from_numpy(s) for i, s in enumerate(source_data)]

        if not os.path.exists(self.cfg.paths.order):
            order = np.arange(len(source_data))
            np.save(self.cfg.paths.order, order)
            warnings.warn(f'Index file `{self.cfg.paths.order}` is created!')
        else:
            order = np.load(self.cfg.paths.order)

        source_data = [source_data[idx] for idx in order]
        self.order = order

        trainset_length = int(self.cfg.series_per_data[0] * self.cfg.train_test_range[0])
        valset_length = int(self.cfg.series_per_data[1] * self.cfg.train_test_range[1])
        testset_length = int(self.cfg.series_per_data[2] * self.cfg.train_test_range[2])
        data_count = trainset_length + valset_length + testset_length

        return {'source': source_data}, data_count

    def get_volume(self, idx):
        if self.cfg.load_mode == 'disk':
            volume = torch.from_numpy(nib.load(self.data['source'][idx]).get_fdata(dtype=np.float32))
        else:
            volume = self.data['source'][idx]
        volume = volume.to(self.cfg.device)
        return volume

    def sample_series(self, points, n, action, rev=False):
        old_dtype = points.dtype
        points = points.type(torch.float64)
        series = [points]
        infos = {}

        count_actions = 1
        infos['count_action'] = count_actions
        infos['actions'] = []
        infos['count_slice_in_action'] = [1]

        for idx in range(count_actions):
            start_points = series[-1]
            normal = utils.simulation.get_normal(series[-1])
            infos['actions'].append(action)

            if action == 'line':
                line_series = None
                while_time = 0
                nn = 7 + np.random.randint(-2, 3)
                while line_series is None:
                    if while_time >= self.cfg.while_max_time:
                        return None
                    while_time += 1
                    direct = utils.simulation.sample_points_by_limit(normal.unsqueeze(0) * (-1 if rev else 1), min_cos=0.98, n=1, dtype=normal.dtype, device=self.cfg.device).squeeze()
                    length = 12 * np.random.rand() + 32
                    end_point = start_points[0, :] + direct * length
                    start_velocity = direct * (2.0 + 0.2 * np.random.randn()) + 0.05 * torch.randn((3,), dtype=direct.dtype, device=self.cfg.device)
                    start_accele = 0
                    line_series = utils.simulation.sample_line(series[-1], end_point, start_velocity, start_accele, radius=self.rad, n=n, min_n=nn - 1)
                line_series = line_series[:nn - 1, ...]
                infos['count_slice_in_action'].append(len(line_series))
                series.extend(line_series)
                action_length = len(line_series)

                line_series = None
                while_time = 0
                nn = 4 + np.random.randint(-2, 2)
                while line_series is None:
                    if while_time >= self.cfg.while_max_time:
                        return None
                    while_time += 1
                    length = 60 * np.random.rand() + 180
                    end_point = end_point + direct * length
                    start_velocity = direct * (18 + 6.0 * np.random.randn()) + 0.05 * torch.randn((3,), dtype=direct.dtype, device=self.cfg.device)
                    start_accele = 0
                    line_series = utils.simulation.sample_line(series[-1], end_point, start_velocity, start_accele, radius=self.rad, n=n, min_n=nn - 1)
                line_series = line_series[:nn - 1, ...]
                infos['count_slice_in_action'][-1] += len(line_series)
                series.extend(line_series)
                action_length += len(line_series)

                line_series = None
                while_time = 0
                nn = 7 + np.random.randint(-2, 3)
                while line_series is None:
                    if while_time >= self.cfg.while_max_time:
                        return None
                    while_time += 1
                    length = 12 * np.random.rand() + 32
                    end_point = end_point + direct * length
                    start_velocity = direct * (2.0 + 0.2 * np.random.randn()) + 0.05 * torch.randn((3,), dtype=direct.dtype, device=self.cfg.device)
                    start_accele = 0
                    line_series = utils.simulation.sample_line(series[-1], end_point, start_velocity, start_accele, radius=self.rad, n=n, min_n=nn - 1)
                line_series = line_series[:nn - 1, ...]
                infos['count_slice_in_action'][-1] += len(line_series)
                series.extend(line_series)
                action_length += len(line_series)

                line_series = None
                while_time = 0
                nn = 4 + np.random.randint(-2, 2)
                while line_series is None:
                    if while_time >= self.cfg.while_max_time:
                        return None
                    while_time += 1
                    length = 60 * np.random.rand() + 180
                    end_point = end_point + direct * length
                    start_velocity = direct * (18 + 6.0 * np.random.randn()) + 0.05 * torch.randn((3,), dtype=direct.dtype, device=self.cfg.device)
                    start_accele = 0
                    line_series = utils.simulation.sample_line(series[-1], end_point, start_velocity, start_accele, radius=self.rad, n=n, min_n=nn - 1)
                line_series = line_series[:nn - 1, ...]
                infos['count_slice_in_action'][-1] += len(line_series)
                series.extend(line_series)
                action_length += len(line_series)

            elif action == 'sector':
                length_edge = torch.norm(start_points[2, :] - start_points[1, :], 2)

                sector_point = start_points[0, :] - (start_points[2, :] - start_points[0, :] + start_points[1, :] - start_points[0, :]) / 2
                sector_series = None
                while_time = 0
                nn = 4 + np.random.randint(-2, 3)
                while sector_series is None:
                    if while_time >= self.cfg.while_max_time:
                        return None
                    while_time += 1
                    pc = start_points[2, :] - start_points[0, :] + start_points[1, :] - start_points[0, :]

                    pc_norm = F.normalize(pc, p=2, dim=0)
                    p1p2 = (start_points[2, :] - start_points[1, :]) / length_edge
                    ed_cos = 1
                    wwhile_time = 0
                    while ed_cos < -0.5 or ed_cos > 0.5:
                        if wwhile_time >= self.cfg.while_max_time:
                            return None
                        wwhile_time += 1
                        min_cos, max_cos = (0.98, 1.0) if rev else (0.93, 0.98)
                        end_direct = utils.simulation.sample_points_by_limit(pc.unsqueeze(0), min_cos=min_cos, max_cos=max_cos, n=1, dtype=pc.dtype, device=self.cfg.device).squeeze()
                        ed = end_direct - torch.sum(end_direct * pc_norm) * pc_norm
                        ed = F.normalize(ed, p=2, dim=0)
                        ed_cos = torch.sum(ed * p1p2)

                    max_velocity = np.pi / 180 * (1.5 + 0.05 * np.random.randn())
                    start_accele = 0
                    end_accele = 0
                    sector_series = utils.simulation.sample_sector(series[-1], sector_point, end_direct, max_velocity, start_accele, end_accele, radius=self.rad, n=n, min_n=nn - 1)
                sector_series = sector_series[:nn - 1, ...]
                infos['count_slice_in_action'].append(len(sector_series))
                series.extend(sector_series)
                action_length = len(sector_series)

                sector_point = start_points[0, :] - (start_points[2, :] - start_points[0, :] + start_points[1, :] - start_points[0, :]) / 2
                sector_series = None
                while_time = 0
                nn = 4 + np.random.randint(-2, 3)
                while sector_series is None:
                    if while_time >= self.cfg.while_max_time:
                        return None
                    while_time += 1
                    pc = start_points[2, :] - start_points[0, :] + start_points[1, :] - start_points[0, :]

                    # TODO more faster, not while
                    pc_norm = F.normalize(pc, p=2, dim=0)
                    p1p2 = (start_points[2, :] - start_points[1, :]) / length_edge
                    ed_cos = 1
                    wwhile_time = 0
                    while ed_cos < -0.5 or ed_cos > 0.5:
                        if wwhile_time >= self.cfg.while_max_time:
                            return None
                        wwhile_time += 1
                        min_cos, max_cos = (0.93, 0.98) if rev else (0.98, 1.0)
                        end_direct = utils.simulation.sample_points_by_limit(pc.unsqueeze(0), min_cos=min_cos, max_cos=max_cos, n=1, dtype=pc.dtype, device=self.cfg.device).squeeze()
                        ed = end_direct - torch.sum(end_direct * pc_norm) * pc_norm
                        ed = F.normalize(ed, p=2, dim=0)
                        ed_cos = torch.sum(ed * p1p2)

                    max_velocity = np.pi / 180 * (1.5 + 0.05 * np.random.randn())
                    start_accele = 0
                    end_accele = 0
                    sector_series = utils.simulation.sample_sector(series[-1], sector_point, end_direct, max_velocity, start_accele, end_accele, radius=self.rad, n=n, min_n=nn - 1)
                sector_series = sector_series[:nn - 1, ...]
                infos['count_slice_in_action'][-1] += len(sector_series)
                series.extend(sector_series)
                action_length += len(sector_series)

            else:
                raise ValueError('{} is not an action type.'.format(action))

        series = torch.stack(series, dim=0)
        infos['count_slices'] = len(series)
        infos['series'] = series.type(old_dtype)
        return infos

    def slice_n(self, idx, n, frame_rate=None, optical_flow=True, edge=True):
        data = self.get_volume(idx)

        points = utils.simulation.sample_points_at_border(n, height=self.cfg.source.height, width=self.cfg.source.width, radius=self.rad, direct_down=self.direct_down)
        series = []

        ii = 0
        while ii < n:
            while_broke_flag = False
            while_broke_count1 = 0
            while_broke_count2 = 0
            rev = False
            has_sector = False
            has_sector2 = False

            ii_loop = 0
            loops = 7
            ss_loop = None

            while ii_loop < loops:
                while_broke_flag = False

                if not has_sector2 and (np.random.rand() < 2 / (loops - 1) or ii_loop == loops - 2):
                    action = 'sector'
                else:
                    if np.random.rand() < 0.33:
                        action = 'sector'
                    else:
                        action = 'line'

                start_point = points[ii] if ii_loop == 0 else ss_loop['series'][-1]
                ss = self.sample_series(start_point, n=None, action=action, rev=rev)
                while_time = 0
                while ss is None:
                    while_time += 1
                    if while_time >= self.cfg.while_max_time:
                        if ii_loop == 0:
                            random_point = utils.simulation.sample_points_at_border(1, height=self.cfg.source.height, width=self.cfg.source.width, radius=self.rad, direct_down=self.direct_down)
                            points[ii] = random_point[0, ...]
                        while_broke_flag = True
                        break
                    rev = not rev
                    ss = self.sample_series(start_point, n=None, action=action, rev=rev)
                if while_broke_flag:
                    while_broke_flag = False
                    rev = not rev
                    while_broke_count1 += 1
                    if while_broke_count1 >= self.cfg.while_max_time:
                        while_broke_count1 = 0
                        while_broke_count2 = 0
                        rev = False
                        has_sector = False
                        has_sector2 = False
                        ii_loop = 0
                        ss_loop = None
                    continue
                if action != 'sector':
                    if frame_rate is not None:
                        ss['series'] = ss['series'][::frame_rate[ii]]
                    ss['count_slices'] = len(ss['series'])
                    ss['count_slice_in_action'][-1] = ss['count_slices'] - 1

                slices = utils.simulation.get_slices(data, ss['series'], height=self.cfg.source.height, width=self.cfg.source.width, volume_origin=self.origin)
                slices_all_value = (torch.norm(slices, 1, dim=[1, 2]) == 0).any()
                w_time = 0
                while slices_all_value:
                    w_time += 1
                    if w_time >= self.cfg.while_max_time:
                        while_broke_flag = True
                        break
                    ss = self.sample_series(start_point, n=None, action=action, rev=rev)
                    while_time = 0
                    while ss is None:
                        while_time += 1
                        if while_time >= self.cfg.while_max_time:
                            if ii_loop == 0:
                                random_point = utils.simulation.sample_points_at_border(1, height=self.cfg.source.height, width=self.cfg.source.width, radius=self.rad, direct_down=self.direct_down)
                                points[ii] = random_point[0, ...]
                            while_broke_flag = True
                            break
                        rev = not rev
                        ss = self.sample_series(start_point, n=None, action=action, rev=rev)
                    if while_broke_flag:
                        break
                    if action != 'sector':
                        if frame_rate is not None:
                            ss['series'] = ss['series'][::frame_rate[ii]]
                        ss['count_slices'] = len(ss['series'])
                        ss['count_slice_in_action'][-1] = ss['count_slices'] - 1
                    slices = utils.simulation.get_slices(data, ss['series'], height=self.cfg.source.height, width=self.cfg.source.width, volume_origin=self.origin)
                    slices_all_value = (torch.norm(slices, 1, dim=[1, 2]) == 0).any()
                if while_broke_flag:
                    while_broke_flag = False
                    rev = not rev
                    while_broke_count2 += 1
                    if while_broke_count2 >= self.cfg.while_max_time:
                        while_broke_count1 = 0
                        while_broke_count2 = 0
                        rev = False
                        has_sector = False
                        has_sector2 = False
                        ii_loop = 0
                        ss_loop = None
                    continue

                ss['slices'] = slices
                if ii_loop == 0:
                    ss_loop = ss
                else:
                    ss_loop['count_slice_in_action'][-1] += ss['count_slice_in_action'][-1]
                    ss_loop['count_slices'] += ss['count_slice_in_action'][-1]
                    ss_loop['series'] = torch.cat([ss_loop['series'], ss['series'][1:, ...]], dim=0)
                    ss_loop['slices'] = torch.cat([ss_loop['slices'], ss['slices'][1:, ...]], dim=0)
                if ss_loop['count_slices'] >= self.cfg.series_min_length:
                    break
                ii_loop += 1

                if action == 'sector':
                    if not has_sector:
                        has_sector = True
                    else:
                        has_sector2 = True

            if not while_broke_flag:
                if frame_rate is not None:
                    ss_loop['frame_rate'] = frame_rate[ii]
                ss_loop['gaps'] = utils.simulation.series_to_dof(ss_loop['series'])
                ss_loop['indices'] = torch.randperm(len(ss_loop['gaps']))
                if optical_flow:
                    ss_loop['opticalflow'] = utils.image.get_optical_flow(ss_loop['slices'], device=self.cfg.device)
                if edge:
                    ss_loop['edge'] = utils.image.get_edge(ss_loop['slices'], device=self.cfg.device)
                if frame_rate is None:
                    assert len(ss_loop['slices']) == len(ss_loop['series']) \
                           and len(ss_loop['series']) == ss_loop['count_slices'] \
                           and ss_loop['count_slices'] == ss_loop['count_slice_in_action'][-1] + 1, \
                        (len(ss_loop['slices']), len(ss_loop['series']), ss_loop['count_slices'], ss_loop['count_slice_in_action'][-1])
                series.append(ss_loop)
                ii += 1

        return series

    def pad(self, data, max_length=None):
        if data is None:
            return None
        max_length = max_length or self.cfg.series_max_length
        if len(data) > max_length:
            data = data[:max_length]
        return data

    def __getitem__(self, index):
        idx_num = self.get_idx(index)
        idx = str(idx_num)

        if idx not in self.pr.keys() or len(self.pr[idx]) == 0:
            ps = self.cfg.ps if index < self.trainset_length else 1
            fr = torch.randint(self.cfg.frame_rate[0], self.cfg.frame_rate[1] + 1, (ps,))
            self.pd[idx] = self.slice_n(idx_num, n=ps, frame_rate=fr)
            self.pr[idx] = torch.randperm(ps)
        data = self.pd[idx][self.pr[idx][0].item()]

        info = torch.tensor([len(data['slices'])])
        frame_rate = data['frame_rate']
        pad_slices = self.pad(data['slices'])
        pad_gaps = self.pad(data['gaps'], self.cfg.series_max_length - 1)
        pad_series = self.pad(data['series'])
        pad_optical_flow = self.pad(data['opticalflow'], self.cfg.series_max_length - 1)
        pad_edge = self.pad(data['edge'])

        source = pad_slices.unsqueeze(1)
        target = torch.cat([F.pad(pad_gaps, (0, 0, 0, 1)), pad_series.view(-1, 9)], dim=-1)

        self.pd[idx][self.pr[idx][0].item()] = []
        self.pr[idx] = self.pr[idx][1:]

        sample_dict = {
            'source': source, 'target': target,
            'optical_flow': pad_optical_flow,
            'edge': pad_edge,
            'frame_rate': frame_rate,
            'info': info
        }

        utils.common.set_seed(int(time.time() * 1000) % (1 << 32) + index)
        return sample_dict, index
