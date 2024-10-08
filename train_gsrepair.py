import argparse
import os
import random

import numpy as np
import wandb
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
from test import eval_psnr
from utils import loss_fn

from torch.utils.data.distributed import DistributedSampler


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True, sampler=sampler
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    train_loader.dataset.set_dynamic_scale_factor()
    i = 0
    last_pred = None
    last_gt = None
    for batch in tqdm(train_loader, leave=False, desc='train'):
        i += 1
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        target_shape = batch['gt'].shape[-2:]
        pred = model(inp, target_shape)
        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_pred = pred.detach()
        last_gt = gt.detach()
        pred = None
        loss = None
        train_loader.dataset.set_dynamic_scale_factor()

    return train_loss.item(), last_pred, last_gt


def main(config_, save_path):
    global config, log, writer
    config = config_

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        log = utils.set_save_path(save_path, config)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)

        train_loss, train_pred, train_gt = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        if local_rank == 0:
            log_info.append('train: loss={:.4f}'.format(train_loss))
            wandb.log({
                'train_loss': train_loss,
                'train_imgs_epoch/pred': wandb.Image(train_pred),
                'train_imgs_epoch/gt': wandb.Image(train_gt)}, step=epoch
            )

        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        if local_rank == 0:
            torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0) and (local_rank == 0):
            torch.save(sv_file, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            val_res, val_pred, val_gt = eval_psnr(
                val_loader, model,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize')
            )

            if local_rank == 0:
                log_info.append('val: psnr={:.4f}'.format(val_res))
                wandb.log({
                    'val_psnr': val_res,
                    'val_imgs_epoch/pred': wandb.Image(val_pred),
                    'val_imgs_epoch/gt': wandb.Image(val_gt)}, step=epoch
                )

                if val_res > max_val_v:
                    max_val_v = val_res
                    torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        if local_rank == 0:
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
            log(', '.join(log_info))


def set_random_seed(seed=42):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)

    args = parser.parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend='nccl')

    set_random_seed()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
