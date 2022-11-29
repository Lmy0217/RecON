import argparse
import os
import time

import numpy as np
import torch

import configs
import datasets
import models
import utils


class Main(object):

    def __init__(self, args):
        self.args = args
        self.model_cfg = configs.BaseConfig(utils.common.real_config_path(
            args.model_config_path, configs.env.paths.model_cfgs_folder))
        self.run_cfg = configs.Run(utils.common.real_config_path(
            args.run_config_path, configs.env.paths.run_cfgs_folder), gpus=args.gpus)
        self.dataset_cfg = datasets.functional.common.more(configs.BaseConfig(
            utils.common.real_config_path(args.dataset_config_path, configs.env.paths.dataset_cfgs_folder)))
        print(args)

        self._init()
        self._get_component()
        self.show_cfgs()

    def _init(self):
        utils.common.set_seed(0)
        self.msg = {}

    def _get_component(self):
        self.path = utils.common.get_path(self.model_cfg, self.dataset_cfg, self.run_cfg)
        self.logger = utils.Logger(self.path, utils.common.get_filename(self.model_cfg._path))

        self.dataset = datasets.functional.common.find(self.dataset_cfg.name)(self.dataset_cfg, logger=self.logger)
        self.model = models.functional.common.find(self.model_cfg.name)(
            self.model_cfg, self.dataset.cfg, self.run_cfg, dataset=self.dataset, logger=self.logger, main_msg=self.msg)
        self.start_epoch = self.model.load(self.args.test_epoch)

    def show_cfgs(self):
        self.logger.info(self.model.cfg)
        self.logger.info(self.run_cfg)
        self.logger.info(self.dataset.cfg)

    def split(self):
        self.trainset, self.valset, self.testset = self.dataset.split()

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.run_cfg.batch_size,
            shuffle=True,
            collate_fn=getattr(self.trainset.dataset, 'collate_fn', None),
            num_workers=self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
            sampler=None
        )

        self.run_cfg.test_batch_size = getattr(self.run_cfg, 'test_batch_size', self.run_cfg.batch_size)
        self.val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.run_cfg.test_batch_size,
            shuffle=False,
            collate_fn=getattr(self.valset.dataset, 'collate_fn', None),
            num_workers=self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
            sampler=None
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.run_cfg.test_batch_size,
            shuffle=False,
            collate_fn=getattr(self.testset.dataset, 'collate_fn', None),
            num_workers=self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
            sampler=None
        )

    def train(self, epoch):
        utils.common.set_seed(int(time.time()) + epoch)
        torch.cuda.empty_cache()
        count, loss_all = 0, {}
        batch_per_epoch, count_data = len(self.train_loader), len(self.train_loader.dataset)
        log_step = 1
        epoch_info = {'epoch': epoch, 'batch_per_epoch': batch_per_epoch, 'count_data': count_data}
        for batch_idx, (sample_dict, index) in enumerate(self.train_loader):
            _count = len(list(sample_dict.values())[0])
            epoch_info['batch_idx'] = batch_idx
            epoch_info['index'] = index
            epoch_info['batch_count'] = _count
            loss_dict = self.model.train(epoch_info, sample_dict)
            loss_dict['_count'] = _count
            utils.common.merge_dict(loss_all, loss_dict)
            count += _count
            if batch_idx % log_step == 0:
                self.logger.info_scalars('Train Epoch: {} [{}/{} ({:.0f}%)]\t', (epoch, count, count_data, 100. * count / count_data), loss_dict)
        if epoch % self.run_cfg.save_step == 0:
            loss_file = os.path.join(self.path, self.model.name + '_' + str(epoch) + configs.env.paths.loss_file)
            self.logger.save_npy(loss_file, {k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for k, v in loss_all.items()})
        loss_all = self.model.train_return_hook(epoch_info, loss_all)
        self.logger.info_scalars('Train Epoch: {}\t', (epoch,), loss_all)
        if epoch % self.run_cfg.save_step == 0:
            self.model.save(epoch)

    def test(self, epoch, data_loader=None, log_text=None):
        utils.common.set_seed(int(time.time()) + epoch)
        torch.cuda.empty_cache()
        predict, count = {}, 0
        data_loader = data_loader or self.test_loader
        log_text = log_text or 'Test'
        with torch.no_grad():
            batch_per_epoch, count_data = len(data_loader), len(data_loader.dataset)
            log_step = max(int(np.power(10, np.floor(np.log10(batch_per_epoch / 10)))), 1) if batch_per_epoch > 0 else 1
            epoch_info = {'epoch': epoch, 'batch_per_epoch': batch_per_epoch, 'count_data': count_data, 'log_text': log_text}
            for batch_idx, (sample_dict, index) in enumerate(data_loader):
                _count = len(list(sample_dict.values())[0])
                epoch_info['batch_idx'] = batch_idx
                epoch_info['index'] = index
                epoch_info['batch_count'] = _count
                output_dict = self.model.test(epoch_info, sample_dict)
                count += _count
                if batch_idx % log_step == 0:
                    self.logger.info('{} Epoch: {} [{}/{} ({:.0f}%)]'.format(log_text, epoch, count, count_data, 100. * count / count_data))
                for name, value in output_dict.items():
                    v = value.float() if value.shape else value.unsqueeze(0)
                    v = v.cpu().numpy()
                    predict[name] = np.concatenate([predict[name], v]) if name in predict.keys() else v
        predict = self.model.test_return_hook(epoch_info, predict)
        predict_file = os.path.join(self.path, self.model.name + '_' + str(epoch) + configs.env.paths.predict_file)
        self.logger.save_npy(predict_file, predict)

    def val_test(self, epoch):
        self.test(epoch, data_loader=self.val_loader, log_text='Val')
        self.test(epoch, data_loader=self.test_loader, log_text='Test')


def run():
    parser = argparse.ArgumentParser(description='RecON')
    parser.add_argument('-m', '--model_config_path', type=str, required=True, metavar='/path/to/model/config.json',
                        help='Path to model config .json file')
    parser.add_argument('-r', '--run_config_path', type=str, required=True, metavar='/path/to/run/config.json',
                        help='Path to run config .json file')
    parser.add_argument('-d', '--dataset_config_path', type=str, required=True, metavar='/path/to/dataset/config.json',
                        help='Path to dataset config .json file')
    parser.add_argument('-g', '--gpus', type=str, default='0', metavar='cuda device, i.e. 0 or cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument('-t', '--test_epoch', type=int, metavar='epoch want to test', help='epoch want to test')
    args = parser.parse_args()

    main = Main(args)
    main.split()
    if args.test_epoch is None:
        if main.start_epoch == 0:
            main.val_test(main.start_epoch)
        for epoch in range(main.start_epoch + 1, main.run_cfg.epochs + 1):
            main.train(epoch)
            if epoch % main.run_cfg.save_step == 0:
                main.val_test(epoch)
    else:
        main.test(main.start_epoch)


if __name__ == '__main__':
    run()
