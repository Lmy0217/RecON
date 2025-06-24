import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import configs
import models
import utils


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.resnet3d = models.layers.resnet3d.generate_model(10, n_input_channels=1, n_classes=1)

    def forward(self, x):
        x = self.resnet3d(x)
        return x


class Online_Discriminator(models.BaseModel):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)
        self.backbone = models.online_backbone.Backbone(self.data_cfg.source.channel, self.data_cfg.target.elements - 9).to(self.device)
        self.backbone.load_state_dict(torch.load(configs.env.getdir(self.cfg.backbone_weight)))
        self.discriminator = Discriminator().to(self.device)

        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.run.lr, betas=self.run.betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.run.step_size, gamma=self.run.gamma)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

        self.down_ratio = 0.3
        self.mat_scale = torch.eye(4, dtype=torch.float32, device=self.device)
        self.mat_scale[0, 0] = self.down_ratio
        self.mat_scale[1, 1] = self.down_ratio
        self.mat_scale[2, 2] = self.down_ratio

    def get_reco(self, epoch_info, real_source, real_target, edge, optical_flow):
        self.backbone.eval()
        with torch.no_grad():
            down_source = F.interpolate(real_source.squeeze(-3), scale_factor=self.down_ratio).unsqueeze(-3)
            real_input = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...], edge[:, :-1, ...], edge[:, 1:, ...], optical_flow], dim=2)
            real_gaps = real_target[:, :-1, :-9]
            real_series_0 = real_target[:, 0, -9:].view(-1, 3, 3)

            fake_gaps, _ = self.backbone(real_input, return_feature=False)
            fake_gaps = torch.cat([fake_gaps[:, :, :3], fake_gaps[:, :, 3:] / 100], dim=-1)

            real_series = utils.simulation.dof_to_series(real_series_0, real_gaps)
            fake_series = utils.simulation.dof_to_series(real_series_0, fake_gaps)

            reco, label = [], []
            for idx, index in enumerate(epoch_info['index']):
                if np.random.rand() >= 0.5:
                    r_reco, _ = utils.reconstruction.reco(down_source[idx].squeeze(0).squeeze(1), real_series[idx].squeeze(0), mat_scale=self.mat_scale)
                    reco.append(r_reco)
                    label.append(1)
                else:
                    r_reco, _ = utils.reconstruction.reco(down_source[idx].squeeze(0).squeeze(1), fake_series[idx].squeeze(0), mat_scale=self.mat_scale)
                    reco.append(r_reco)
                    label.append(0)
                torch.cuda.empty_cache()
            reco = torch.stack(reco, dim=0).unsqueeze(1)
            label = torch.tensor(label, dtype=torch.float32, device=self.device)

        return reco, label

    def train(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device)
        edge = sample_dict['edge'].to(self.device)
        optical_flow = sample_dict['optical_flow'].to(self.device)

        self.discriminator.train()
        self.optimizer.zero_grad()
        reco, label = self.get_reco(epoch_info, real_source, real_target, edge, optical_flow)
        pred = self.discriminator(reco)
        loss = self.criterion(pred, label.unsqueeze(-1))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(epoch_info['epoch'])

        return {'loss': loss}

    def test(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device)
        edge = sample_dict['edge'].to(self.device)
        optical_flow = sample_dict['optical_flow'].to(self.device)

        self.discriminator.eval()
        reco, label = self.get_reco(epoch_info, real_source, real_target, edge, optical_flow)
        pred = self.discriminator(reco)
        pred_label = pred >= 0
        accuracy = pred_label.eq(label.view_as(pred_label)).sum().item() / len(reco)
        accuracy = torch.tensor(accuracy, dtype=torch.float32, device=self.device)

        return {'accuracy': accuracy}

    def test_return_hook(self, epoch_info, return_all):
        return_info = {}
        for key, value in return_all.items():
            return_info[key] = np.sum(value) / epoch_info['batch_per_epoch']
        if return_info:
            self.logger.info_scalars('{} Epoch: {}\t', (epoch_info['log_text'], epoch_info['epoch']), return_info)
        return return_all
