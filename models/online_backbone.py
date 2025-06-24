import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils


class Backbone(nn.Module):

    def __init__(self, in_planes, num_classes):
        super().__init__()
        self.resnet = timm.create_model('resnet18', pretrained=True, in_chans=in_planes, num_classes=0, global_pool='')
        self.lstm = models.layers.convolutional_rnn.Conv2dLSTM(512, 512, kernel_size=3, batch_first=True)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feature=False):
        b, t, c, h, w = x.shape
        x = (x - torch.mean(x, dim=[3, 4], keepdim=True)) / (torch.std(x, dim=[3, 4], keepdim=True) + 1e-6)
        x = x.view(b * t, c, h, w)
        x = self.resnet(x)
        x = x.view(b, t, *x.shape[1:])
        if return_feature:
            f = self.avg(x)
            f = f.view(f.size(0), f.size(1), -1)
        else:
            f = None
        x = self.lstm(x)[0]
        x = self.avg(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc(x)
        return x, f


class Online_Backbone(models.BaseModel):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)
        self.backbone = Backbone(self.data_cfg.source.channel, self.data_cfg.target.elements - 9).to(self.device)
        self.optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.run.lr, betas=self.run.betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.run.step_size, gamma=self.run.gamma)
        self.flag_motion = True

    def criterion(self, real_target, fake_target, feature=None):
        real_dist, real_angle = real_target.split([3, self.data_cfg.target.elements - 12], dim=-1)
        fake_dist, fake_angle = fake_target.split([3, self.data_cfg.target.elements - 12], dim=-1)

        loss_dist = F.l1_loss(real_dist, fake_dist) * 3
        loss_angle = F.l1_loss(real_angle, fake_angle) * 3
        loss_corr = utils.metric.correlation_loss(real_target, fake_target)

        loss_dict = {'loss_dist': loss_dist, 'loss_angle': loss_angle, 'loss_corr': loss_corr}

        if self.flag_motion:
            fake_motion = torch.norm(fake_dist, p=2, dim=-1) + 1e-6
            feature = torch.norm(feature, p=2, dim=-1)
            loss_motion = torch.mean(feature / fake_motion) * self.cfg.weight_motion
            loss_dict['loss_motion'] = loss_motion

        return loss_dict

    def train(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device)
        edge = sample_dict['edge'].to(self.device)
        optical_flow = sample_dict['optical_flow'].to(self.device)

        real_target = real_target[:, :-1, :-9]
        real_target[:, :, 3:] = real_target[:, :, 3:] * 100

        self.backbone.train()
        self.optimizer.zero_grad()
        input = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...], edge[:, :-1, ...], edge[:, 1:, ...], optical_flow], dim=2)
        fake_target, feature = self.backbone(input, return_feature=self.flag_motion)

        losses = self.criterion(real_target, fake_target, feature)
        loss = sum(losses.values())
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(epoch_info['epoch'])

        return {'loss': loss, **losses}

    def test(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device).squeeze(0)
        edge = sample_dict['edge'].to(self.device)
        optical_flow = sample_dict['optical_flow'].to(self.device)

        real_series = real_target[:, -9:].view(-1, 3, 3)

        self.backbone.eval()
        input = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...], edge[:, :-1, ...], edge[:, 1:, ...], optical_flow], dim=2)
        fake_gaps, _ = self.backbone(input)
        fake_gaps = fake_gaps[0, :, :]
        fake_gaps[:, 3:] /= 100

        fake_series = utils.simulation.dof_to_series(real_series[0:1, :, :], fake_gaps.unsqueeze(0)).squeeze(0)
        losses = utils.metric.get_metric(real_series, fake_series)

        return losses

    def test_return_hook(self, epoch_info, return_all):
        return_info = {}
        for key, value in return_all.items():
            return_info[key] = np.sum(value) / epoch_info['batch_per_epoch']
        if return_info:
            self.logger.info_scalars('{} Epoch: {}\t', (epoch_info['log_text'], epoch_info['epoch']), return_info)
        return return_all
