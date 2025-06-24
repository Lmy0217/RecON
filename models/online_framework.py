import os

import numpy as np
import torch
import torch.nn.functional as F

import configs
import models
import utils


def pad_volume(v1, v2):
    max_s0 = max(v1.shape[0], v2.shape[0])
    max_s1 = max(v1.shape[1], v2.shape[1])
    max_s2 = max(v1.shape[2], v2.shape[2])
    diff_v1_s0 = max_s0 - v1.shape[0]
    diff_v1_s1 = max_s1 - v1.shape[1]
    diff_v1_s2 = max_s2 - v1.shape[2]
    diff_v2_s0 = max_s0 - v2.shape[0]
    diff_v2_s1 = max_s1 - v2.shape[1]
    diff_v2_s2 = max_s2 - v2.shape[2]
    v1 = F.pad(v1, (diff_v1_s2 // 2, diff_v1_s2 - diff_v1_s2 // 2, diff_v1_s1 // 2, diff_v1_s1 - diff_v1_s1 // 2, diff_v1_s0 // 2, diff_v1_s0 - diff_v1_s0 // 2))
    v2 = F.pad(v2, (diff_v2_s2 // 2, diff_v2_s2 - diff_v2_s2 // 2, diff_v2_s1 // 2, diff_v2_s1 - diff_v2_s1 // 2, diff_v2_s0 // 2, diff_v2_s0 - diff_v2_s0 // 2))
    return v1, v2


class Online_Framework(models.BaseModel):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)
        self.backbone = models.online_backbone.Backbone(self.data_cfg.source.channel, self.data_cfg.target.elements - 9).to(self.device)
        self.backbone_start_weight = torch.load(configs.env.getdir(self.cfg.backbone_weight))
        self.backbone.load_state_dict(self.backbone_start_weight)

        self.discriminator = models.online_discriminator.Discriminator().to(self.device)
        self.discriminator_start_weight = torch.load(configs.env.getdir(self.cfg.discriminator_weight))
        self.discriminator.load_state_dict(self.discriminator_start_weight)

        self.mat_scale = torch.eye(4, dtype=torch.float32, device=self.device)
        self.mat_scale[0, 0] = self.cfg.down_ratio
        self.mat_scale[1, 1] = self.cfg.down_ratio
        self.mat_scale[2, 2] = self.cfg.down_ratio

    def train(self, epoch_info, sample_dict):
        return {}

    def criterion(self, real_target, fake_target):
        real_dist, real_angle = real_target.split([3, self.data_cfg.target.elements - 12], dim=-1)
        fake_dist, fake_angle = fake_target.split([3, self.data_cfg.target.elements - 12], dim=-1)

        loss_dist = F.l1_loss(real_dist, fake_dist) * 3
        loss_angle = F.l1_loss(real_angle, fake_angle) * 3
        loss_corr = utils.metric.correlation_loss(real_target, fake_target)

        loss_dict = {'loss_dist': loss_dist, 'loss_angle': loss_angle, 'loss_corr': loss_corr}
        return loss_dict

    def test_optimize(self, epoch_info, real_source, real_target, edge, optical_flow, epoch):
        self.backbone.load_state_dict(self.backbone_start_weight)
        self.discriminator.load_state_dict(self.discriminator_start_weight)

        down_source = F.interpolate(real_source.squeeze(-3), scale_factor=self.cfg.down_ratio).unsqueeze(-3)
        real_gaps = real_target[0, :-1, :-9]
        real_series = real_target[0, :, -9:].view(-1, 3, 3)

        real_input = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...], edge[:, :-1, ...], edge[:, 1:, ...], optical_flow], dim=2)

        value = {'real_source': real_source, 'real_gaps': real_gaps.clone(), 'real_series': real_series.clone()}
        self.backbone.eval()
        self.discriminator.eval()
        fake2_gaps, _ = self.backbone(real_input, return_feature=False)
        fake2_gaps = fake2_gaps[0, :, :]
        fake2_gaps[:, 3:] /= 100
        fake2_series = utils.simulation.dof_to_series(real_series[0:1, :, :], fake2_gaps.unsqueeze(0)).squeeze(0)
        losses2 = utils.metric.get_metric(real_series, fake2_series)
        value['fake_gaps'] = [fake2_gaps]
        value['fake_series'] = [fake2_series]
        value['loss'] = [losses2]

        self.optimizer_psc = torch.optim.Adam(self.backbone.parameters(), lr=self.run.lr_psc, betas=self.run.betas)
        self.optimizer_g = torch.optim.Adam(self.backbone.parameters(), lr=self.run.lr_fcc_gas, betas=self.run.betas)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.run.lr_fcc_gas, betas=self.run.betas)

        with torch.enable_grad():
            for idx in range(1, epoch + 1):
                self.logger.info(f"RecON: Data {epoch_info['index'].item() + 1}/{epoch_info['count_data']} Epoch {idx}/{epoch}")
                self.backbone.train()
                self.discriminator.train()

                with torch.no_grad():
                    fake_gaps, _ = self.backbone(real_input, return_feature=False)
                    fake_gaps = torch.cat([fake_gaps[:, :, :3], fake_gaps[:, :, 3:] / 100], dim=-1)

                for idx_psc in range(1, self.cfg.psc_epoch + 1):
                    with torch.no_grad():
                        corr = np.Inf
                        acq = 0
                        best_corr = np.Inf
                        best_sample = None
                        while corr > 1 - self.cfg.psc_threshold and acq < self.cfg.psc_max_acquisition:
                            d_idx = torch.randint(self.dataset.trainset_length, (1,), dtype=torch.long, device=self.device)
                            r_data = self.dataset[d_idx[0]][0]
                            gaps = r_data['target'].to(self.device).unsqueeze(0)
                            gaps = gaps[:, :-1, :-9]
                            min_length = min(fake_gaps.shape[1], gaps.shape[1])
                            corr = utils.metric.correlation_loss(fake_gaps[:, :min_length], gaps[:, :min_length])
                            if corr < best_corr:
                                best_corr = corr
                                best_sample = r_data, gaps
                            acq += 1
                        if acq == self.cfg.psc_max_acquisition:
                            r_data, gaps = best_sample
                        slices = r_data['source'].to(self.device).unsqueeze(0)
                        ed = r_data['edge'].to(self.device).unsqueeze(0)
                        of = r_data['optical_flow'].to(self.device).unsqueeze(0)
                        gaps[:, :, 3:] = gaps[:, :, 3:] * 100

                    self.optimizer_psc.zero_grad()
                    ri = torch.cat([slices[:, :-1, ...], slices[:, 1:, ...], ed[:, :-1, ...], ed[:, 1:, ...], of], dim=2)
                    fgaps, feature = self.backbone(ri, return_feature=False)

                    losses = self.criterion(gaps, fgaps)
                    loss = sum(losses.values())

                    self.logger.info_scalars('PSC iter {}/{}\t', (idx_psc, self.cfg.psc_epoch), {'loss_psc': loss, **losses})
                    loss.backward()
                    self.optimizer_psc.step()
                torch.cuda.empty_cache()

                if idx % self.cfg.discriminator_opt_cycle == 0:
                    with torch.no_grad():
                        d_idx = torch.randint(self.dataset.trainset_length, (1,), dtype=torch.long, device=self.device)
                        r_data = self.dataset[d_idx[0]][0]
                        r_source = r_data['source'].to(self.device)
                        r_target = r_data['target'].to(self.device)

                        r_down = F.interpolate(r_source.unsqueeze(0).squeeze(-3), scale_factor=self.cfg.down_ratio).unsqueeze(-3)
                        r_reco, _ = utils.reconstruction.reco(r_down[:, ::2].squeeze(0).squeeze(1), r_target[::2, -9:].view(-1, 3, 3), mat_scale=self.mat_scale)

                    self.optimizer_d.zero_grad()

                    fake_gaps, _ = self.backbone(real_input, return_feature=False)
                    fake_gaps = torch.cat([fake_gaps[0, :, :3], fake_gaps[0, :, 3:] / 100], dim=-1)
                    fake_series = utils.simulation.dof_to_series(real_series[0:1, :, :], fake_gaps.unsqueeze(0)).squeeze(0)

                    reco, _ = utils.reconstruction.reco(down_source[:, ::2].squeeze(0).squeeze(1), fake_series[::2], mat_scale=self.mat_scale)

                    pred_real = self.discriminator(r_reco.unsqueeze(0).unsqueeze(0))
                    pred_fake = self.discriminator(reco.unsqueeze(0).unsqueeze(0))
                    loss_d_gas = pred_fake - pred_real
                    r_reco_resize, reco_resize = pad_volume(r_reco, reco)
                    d_norm = 2 * (r_reco_resize - reco_resize).abs().mean()
                    loss_qp = loss_d_gas ** 2 / d_norm
                    loss_d = loss_d_gas + loss_qp
                    loss_d = torch.mean(loss_d)

                    self.logger.info_scalars('GAS_d\t\t', (), {'loss_d': loss_d.item(), 'loss_d_gas': loss_d_gas.item(), 'loss_qp': loss_qp.item()})
                    loss_d.backward()
                    self.optimizer_d.step()
                torch.cuda.empty_cache()

                self.optimizer_g.zero_grad()

                fake_gaps, _ = self.backbone(real_input, return_feature=False)
                fake_gaps = torch.cat([fake_gaps[0, :, :3], fake_gaps[0, :, 3:] / 100], dim=-1)
                fake_series = utils.simulation.dof_to_series(real_series[0:1, :, :], fake_gaps.unsqueeze(0)).squeeze(0)

                index_rate = self.cfg.reco_rate
                index_reco = (torch.arange(int(fake_series.shape[0] * index_rate), device=self.device) / index_rate).type(torch.int64)
                index_slice = torch.tensor([i for i in range(fake_series.shape[0]) if i not in index_reco], device=self.device)
                reco, min_point = utils.reconstruction.reco(down_source.index_select(1, index_reco).squeeze(0).squeeze(1), fake_series.index_select(0, index_reco), mat_scale=self.mat_scale)
                r_rec = F.interpolate(reco.unsqueeze(0).unsqueeze(0), scale_factor=1 / self.cfg.down_ratio).squeeze(0).squeeze(0)
                slices = utils.reconstruction.get_slice(r_rec, fake_series.index_select(0, index_slice) - min_point.unsqueeze(0).unsqueeze(0), real_source.shape[-2:])

                loss_fcc = F.l1_loss(slices, real_source.index_select(1, index_slice)) * self.cfg.weight_fcc
                pred_fake = self.discriminator(reco.unsqueeze(0).unsqueeze(0))
                loss_g_gas = -torch.mean(pred_fake)
                loss_g = loss_g_gas + loss_fcc

                self.logger.info_scalars('GAS_g+FCC\t', (), {'loss_g': loss_g.item(), 'loss_g_gas': loss_g_gas.item(), 'loss_fcc': loss_fcc.item()})
                loss_g.backward()
                self.optimizer_g.step()

                with torch.no_grad():
                    self.backbone.eval()
                    self.discriminator.eval()
                    fake2_gaps, _ = self.backbone(real_input, return_feature=False)
                    fake2_gaps = fake2_gaps[0, :, :]
                    fake2_gaps[:, 3:] /= 100
                    fake2_series = utils.simulation.dof_to_series(real_series[0:1, :, :], fake2_gaps.unsqueeze(0)).squeeze(0)
                    losses2 = utils.metric.get_metric(real_series, fake2_series)
                    value['fake_gaps'].append(fake2_gaps)
                    value['fake_series'].append(fake2_series)
                    value['loss'].append(losses2)

        self.backbone.eval()
        self.discriminator.eval()
        return value

    def test(self, epoch_info, sample_dict):
        if epoch_info['index'].item() < 0:
           return {}
        utils.common.set_seed(epoch_info['index'].item() * 42)
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device)
        edge = sample_dict['edge'].to(self.device)
        optical_flow = sample_dict['optical_flow'].to(self.device)
        frame_rate = sample_dict['frame_rate']
        length = min(sample_dict['info'], real_source.shape[1])

        value = self.test_optimize(epoch_info, real_source, real_target, edge, optical_flow, epoch=self.run.ol_epochs)
        value['frame_rate'] = [frame_rate]
        value['length'] = [length]

        path = os.path.join(self.path, 'RecON')
        if not os.path.exists(path):
            os.makedirs(path)
        source = value.pop('real_source')
        torch.save(source, os.path.join(path, 'source_' + str(epoch_info['index'].item()) + '.npy'))
        torch.save(value, os.path.join(path, 'value_' + str(epoch_info['index'].item()) + '.npy'))

        return {}
