python imagenet.py -a resnet50 --data /home/severance/ImageNet --epochs 50 --schedule 30 60 90 --checkpoint inference/ --gpu-id 0,1 --train-batch 128 --lr 0.2 --mixbn

'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import numpy as np
from PIL import ImageFile

from FGSM import fgsm_attack
from augmentations.cutmix import cutmix_data
from augmentations.mixup import mixup_data, mixup_criterion
from aux_bn import MixBatchNorm2d, to_mix_status, to_clean_status, to_adv_status
from utils.lr_scheduler import WarmUpLR, adjust_learning_rate

ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import math
import os
import shutil
import time
import random
from functools import partial

# pytorch related
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
from torch.optim.lr_scheduler import _LRScheduler

import models.imagenet as customized_models
from models.AdaIN import StyleTransfer

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.imagenet_a import indices_in_1k
from tensorboardX import SummaryWriter

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--warm', default=5, type=int,
                    help='# of warm up epochs')
parser.add_argument('--warm_lr', default=0., type=float,
                    help='warm up start learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='/tmp/checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--load', default='', type=str,
                    help='load the checkpoint for finetune / evaluation')
parser.add_argument('--finetune', action='store_true',
                    help='ignore aux bn when finetune')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29,
                    help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32,
                    help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4,
                    help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4,
                    help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of classes')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default='7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Core of debiased training
parser.add_argument('--style', action='store_true',
                    help='use style transfer')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='alpha value for style transfer')
parser.add_argument('--label-gamma', default=0.8, type=float,
                    help='gamma in Eq. (1) in paper')
parser.add_argument('--mixbn', action='store_true',
                    help='whether using auxiliary batch normalization')
parser.add_argument('--lr_schedule', type=str, default='step', choices=['step', 'cos'])
parser.add_argument('--multi_grid', action='store_true',
                    help='use downsampled images as input of style transfer for speed up training process')
parser.add_argument('--min_size', default=112, type=int,
                    help='the min size of down sampled images')

# Combine with other data augmentations
parser.add_argument('--mixup', default=0., type=float,
                    help='mixup hyper-parameter')
parser.add_argument('--cutmix', default=0., type=float,
                    help='cutmix hyper-parameter')

# Evaluation options
parser.add_argument('--evaluate_imagenet_c', action='store_true',
                    help="for evaluate Imagenet-C")
parser.add_argument('--already224', action='store_true',
                    help="skip crop and resize if inputs are already 224x224 (for evaluate Stylized-ImageNet)")
parser.add_argument('--imagenet-a', action='store_true',
                    help="mapping the 1k labels to 200 labels (for evaluate ImageNet-A)")
parser.add_argument('--FGSM', action='store_true',
                    help="evalute FGSM robustness")

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc, state
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,  # normalization should be after style transfer module
    ])
    train_dataset = datasets.ImageFolder(traindir, transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_batch, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    val_transforms = [
        transforms.ToTensor(),
        normalize,
    ]
    if not args.already224:
        # This option is for evaluating Stylized-ImageNet, which is already 224x224
        val_transforms = [transforms.Scale(256), transforms.CenterCrop(224)] + val_transforms
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(val_transforms)),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True) if not args.evaluate_imagenet_c else None

    # create model
    if args.arch.startswith('resnext'):
        norm_layer = MixBatchNorm2d if args.mixbn else None
        model = models.__dict__[args.arch](
            baseWidth=args.base_width,
            cardinality=args.cardinality,
            num_classes=args.num_classes,
            norm_layer=norm_layer
        )
    else:
        assert args.arch.startswith('resnet')
        print("=> creating model '{}'".format(args.arch))
        if args.mixbn:
            norm_layer = MixBatchNorm2d
        else:
            norm_layer = None
        # model = models.__dict__[args.arch](num_classes=args.num_classes, norm_layer=norm_layer)
        t_model = torchvision.models.resnet50(num_classes=args.num_classes, norm_layer=norm_layer)
        from models.infodrop_resnet import infodrop_resnet50
        # s_model = torchvision.models.resnet50(num_classes=args.num_classes, norm_layer=norm_layer)
        s_model = infodrop_resnet50(num_classes=args.num_classes, norm_layer=norm_layer)
        # from models.resnet import resnet50
        # s_model = torchvision.models.resnet50(num_classes=args.num_classes, norm_layer=norm_layer)


    from models.cps_network import CPSNetwork
    network = CPSNetwork(s_model=s_model, t_model=t_model)
    # model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.DataParallel(network).cuda()


    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume training
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        # load checkpoint
        if args.load:
            checkpoint = torch.load(args.load)
            if 'state_dict' not in checkpoint:  # for loading cutmix resnext101 model
                raw_ckpt = checkpoint
                checkpoint = {'state_dict': raw_ckpt}

            already_mixbn = False
            for key in checkpoint['state_dict']:
                if 'aux_bn' in key:
                    already_mixbn = True
                    break

            if args.mixbn and not already_mixbn:
                to_merge = {}
                for key in checkpoint['state_dict']:
                    if 'bn' in key:
                        tmp = key.split("bn")
                        aux_key = tmp[0] + 'bn' + tmp[1][0] + '.aux_bn' + tmp[1][1:]
                        to_merge[aux_key] = checkpoint['state_dict'][key]
                    elif 'downsample.1' in key:
                        tmp = key.split("downsample.1")
                        aux_key = tmp[0] + 'downsample.1.aux_bn' + tmp[1]
                        to_merge[aux_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'].update(to_merge)

            if args.finetune:
                # models with aux_bn -> models with main bn
                model_state = model.state_dict()
                pretrained_state = {}
                for k, v in checkpoint['state_dict'].items():
                    if k in model_state and v.size() == model_state[k].size():
                        pretrained_state[k] = v
                    else:
                        assert 'fc' in k or 'aux_bn' in k, k
                        if 'fc' in k:
                            pretrained_state[k] = model_state[k]
                checkpoint['state_dict'] = pretrained_state

            model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'S_Train Loss', 'T_Train Loss', 'S_Valid Loss', 'T_Valid Loss', 'S_Train Acc.', 'T_Train Acc.', 'S_Valid Acc.', 'T_Valid Acc.'])

    if args.finetune:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.module.fc.parameters():
            param.requires_grad = True

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda, args.FGSM)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    if args.evaluate_imagenet_c:
        print("Evaluate ImageNet C")
        distortions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
            'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
        ]

        error_rates = []
        for distortion_name in distortions:
            rate = show_performance(distortion_name, model, criterion, start_epoch, use_cuda)
            error_rates.append(rate)
            print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))
        print(distortions)
        print(error_rates)
        print(np.mean(error_rates))
        return

    # Train and val
    writer = SummaryWriter(log_dir=args.checkpoint)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * args.warm,
                                start_lr=args.warm_lr) if args.warm > 0 else None
    for epoch in range(start_epoch, args.epochs):
        if epoch >= args.warm and args.lr_schedule == 'step':
            adjust_learning_rate(optimizer, epoch, args)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[-1]['lr']))

        style_transfer = partial(StyleTransfer(), alpha=args.alpha,
                                 label_mix_alpha=1 - args.label_gamma) if args.style else None
        train_func = partial(train, train_loader=train_loader, model=model, criterion=criterion,
                             optimizer=optimizer, epoch=epoch, use_cuda=use_cuda,
                             warmup_scheduler=warmup_scheduler, mixbn=args.mixbn,
                             style_transfer=style_transfer, writer=writer)
        if args.mixbn:
            model.apply(to_mix_status)
            ((s_train_loss, s_cps_loss, s_train_acc, s_loss_main, s_loss_aux, s_top1_main, s_top1_aux),
            (t_train_loss, t_cps_loss, t_train_acc, t_loss_main, t_loss_aux, t_top1_main, t_top1_aux)) = train_func()
        else:
            (s_train_loss, s_train_acc), (t_train_loss, t_train_acc) = train_func()
        writer.add_scalar('Train/loss_s', s_train_loss, epoch)
        writer.add_scalar('Train/loss_cps_s', s_cps_loss, epoch)
        writer.add_scalar('Train/acc_s', s_train_acc, epoch)
        writer.add_scalar('Train/loss_t', t_train_loss, epoch)
        writer.add_scalar('Train/loss_cps_t', t_cps_loss, epoch)
        writer.add_scalar('Train/acc_t', t_train_acc, epoch)

        if args.mixbn:
            writer.add_scalar('Train/loss_main_s', s_loss_main, epoch)
            writer.add_scalar('Train/loss_aux_s', s_loss_aux, epoch)
            writer.add_scalar('Train/acc_main_s', s_top1_main, epoch)
            writer.add_scalar('Train/acc_aux_s', s_top1_aux, epoch)

            writer.add_scalar('Train/loss_main_t', t_loss_main, epoch)
            writer.add_scalar('Train/loss_aux_t', t_loss_aux, epoch)
            writer.add_scalar('Train/acc_main_t', t_top1_main, epoch)
            writer.add_scalar('Train/acc_aux_t', t_top1_aux, epoch)
            model.apply(to_clean_status)
        s_test_loss, s_test_acc = test(val_loader, model, 1, criterion, epoch, use_cuda)
        t_test_loss, t_test_acc = test(val_loader, model, 2, criterion, epoch, use_cuda)
        writer.add_scalar('Test/loss_s', s_test_loss, epoch)
        writer.add_scalar('Test/acc_s', s_test_acc, epoch)
        writer.add_scalar('Test/loss_t', t_test_loss, epoch)
        writer.add_scalar('Test/acc_s', t_test_acc, epoch)

        # append logger file
        logger.append([state['lr'], s_train_loss, t_train_loss, s_test_loss, t_test_loss, s_train_acc, t_train_acc, s_test_acc, t_test_acc])

        # save model
        test_acc = max(s_test_acc, t_test_acc)
        is_shape = False
        if test_acc == s_test_acc:
            is_shape = True
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, is_shape, checkpoint=args.checkpoint)

    print('Best acc:')
    print(best_acc)
    writer.close()
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def img_size_scheduler(batch_idx, epoch, schedule):
    ret_size = 224.0
    if batch_idx % 2 == 0:
        ret_size /= 2 ** 0.5
    schedule = [0] + schedule + [args.epochs]
    for i in range(len(schedule) - 1):
        if schedule[i] <= epoch < schedule[i + 1]:
            if epoch < (schedule[i] + schedule[i + 1]) / 2:
                ret_size /= 2 ** 0.5
                ri = i
            break
    ret_size = max(int(ret_size + 0.5), args.min_size)
    if batch_idx == 0:
        print("img size is ", ret_size)
    return ret_size, ret_size


def train(train_loader, model, criterion, optimizer, epoch, use_cuda, warmup_scheduler, mixbn=False,
          style_transfer=None, writer=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    s_losses = AverageMeter()
    t_losses = AverageMeter()
    s_cps_losses = AverageMeter()
    t_cps_losses = AverageMeter()
    s_top1 = AverageMeter()
    t_top1 = AverageMeter()
    s_top5 = AverageMeter()
    t_top5 = AverageMeter()
    if mixbn:
        s_losses_main = AverageMeter()
        s_losses_aux = AverageMeter()
        s_top1_main = AverageMeter()
        s_top1_aux = AverageMeter()
        t_losses_main = AverageMeter()
        t_losses_aux = AverageMeter()
        t_top1_main = AverageMeter()
        t_top1_aux = AverageMeter()
    end = time.time()

    MEAN = torch.tensor([0.485, 0.456, 0.406]).cuda()
    STD = torch.tensor([0.229, 0.224, 0.225]).cuda()
    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if epoch < args.warm:
            warmup_scheduler.step()
        elif args.lr_schedule == 'cos':
            adjust_learning_rate(optimizer, epoch, args, batch=batch_idx, nBatch=len(train_loader))

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        if style_transfer is not None:
            if args.multi_grid:
                img_size = img_size_scheduler(batch_idx, epoch, args.schedule)
                resized_inputs = torch.nn.functional.interpolate(inputs, size=img_size)
                inputs_aux, targets_aux = style_transfer(resized_inputs, targets, replace=True)
                inputs = (inputs, inputs_aux)
                if len(targets_aux) == 3:
                    n = targets.size(0)
                    targets = (torch.cat([targets, targets_aux[0]]),
                               torch.cat([torch.zeros(n, dtype=torch.long).cuda(), targets_aux[1]]),
                               torch.cat([torch.zeros(n, dtype=torch.float).cuda(), targets_aux[2]]))
                else:
                    targets = torch.cat([targets, targets_aux])
                if args.mixup:
                    assert not args.cutmix
                    inputs, targets = mixup_data(inputs, targets, alpha=args.mixup, half=True)
                elif args.cutmix:
                    inputs, targets = cutmix_data(inputs, targets, beta=args.cutmix, half=True)
            else:
                inputs, targets = style_transfer(inputs, targets, replace=False)
                if args.mixup:
                    assert not args.cutmix
                    inputs, targets = mixup_data(inputs, targets, alpha=args.mixup, half=True)
                elif args.cutmix:
                    inputs, targets = cutmix_data(inputs, targets, beta=args.cutmix, half=True)
        else:
            if args.mixup:
                inputs, targets = mixup_data(inputs, targets, alpha=args.mixup, half=False)
            elif args.cutmix:
                inputs, targets = cutmix_data(inputs, targets, beta=args.cutmix, half=False)

        if not args.multi_grid:
            inputs = (inputs - MEAN[:, None, None]) / STD[:, None, None]
            s_outputs = model(inputs, step=1)
            t_outputs = model(inputs, step=2)
        else:
            inputs = ((inputs[0] - MEAN[:, None, None]) / STD[:, None, None],
                      (inputs[1] - MEAN[:, None, None]) / STD[:, None, None])
            if args.mixbn:
                model.apply(to_clean_status)
            s_outputs1 = model(inputs[0], step=1)
            t_outputs1 = model(inputs[0], step=2)
            if args.mixbn:
                model.apply(to_adv_status)
            s_outputs2 = model(inputs[1], step=1)
            t_outputs2 = model(inputs[1], step=2)
            s_outputs = torch.cat([s_outputs1, s_outputs2]) # model output
            t_outputs = torch.cat([t_outputs1, t_outputs2]) # model output

        if len(targets) == 3:
            s_loss = mixup_criterion(criterion, s_outputs, targets[0], targets[1], targets[2]).mean()
            t_loss = mixup_criterion(criterion, t_outputs, targets[0], targets[1], targets[2]).mean()
            targets = targets[0]
            s_pseudo_label = s_outputs.argmax(axis=1)
            t_pseudo_label = t_outputs.argmax(axis=1)
            s_cps_loss = criterion(s_outputs, t_pseudo_label).mean()
            t_cps_loss = criterion(t_outputs, s_pseudo_label).mean()
        else:
            s_loss = criterion(s_outputs, targets).mean()
            t_loss = criterion(t_outputs, targets).mean()
            s_pseudo_label = s_outputs.argmax(axis=1)
            t_pseudo_label = t_outputs.argmax(axis=1)
            s_cps_loss = criterion(s_outputs, t_pseudo_label).mean()
            t_cps_loss = criterion(t_outputs, s_pseudo_label).mean()

        # measure accuracy and record loss
        s_prec1, s_prec5 = accuracy(s_outputs.data, targets.data, topk=(1, 5))
        t_prec1, t_prec5 = accuracy(t_outputs.data, targets.data, topk=(1, 5))
        s_losses.update(s_loss.item(), s_outputs.size(0))
        t_losses.update(t_loss.item(), t_outputs.size(0))
        s_cps_losses.update(s_cps_loss.item(), s_outputs.size(0))
        t_cps_losses.update(t_cps_loss.item(), t_outputs.size(0))
        s_top1.update(s_prec1.item(), s_outputs.size(0))
        t_top1.update(t_prec1.item(), t_outputs.size(0))
        s_top5.update(s_prec5.item(), s_outputs.size(0))
        t_top5.update(t_prec5.item(), t_outputs.size(0))

        if mixbn:
            with torch.no_grad():
                batch_size = s_outputs.size(0)
                s_loss_main = criterion(s_outputs[:batch_size // 2], targets[:batch_size // 2]).mean()
                s_loss_aux = criterion(s_outputs[batch_size // 2:], targets[batch_size // 2:]).mean()
                s_prec1_main = accuracy(s_outputs.data[:batch_size // 2],
                                      targets.data[:batch_size // 2], topk=(1,))[0]
                s_prec1_aux = accuracy(s_outputs.data[batch_size // 2:],
                                     targets.data[batch_size // 2:], topk=(1,))[0]

                t_loss_main = criterion(t_outputs[:batch_size // 2], targets[:batch_size // 2]).mean()
                t_loss_aux = criterion(t_outputs[batch_size // 2:], targets[batch_size // 2:]).mean()
                t_prec1_main = accuracy(t_outputs.data[:batch_size // 2],
                                      targets.data[:batch_size // 2], topk=(1,))[0]
                t_prec1_aux = accuracy(t_outputs.data[batch_size // 2:],
                                     targets.data[batch_size // 2:], topk=(1,))[0]

            s_losses_main.update(s_loss_main.item(), batch_size // 2)
            s_losses_aux.update(s_loss_aux.item(), batch_size // 2)
            s_top1_main.update(s_prec1_main.item(), batch_size // 2)
            s_top1_aux.update(s_prec1_aux.item(), batch_size // 2)

            t_losses_main.update(t_loss_main.item(), batch_size // 2)
            t_losses_aux.update(t_loss_aux.item(), batch_size // 2)
            t_top1_main.update(t_prec1_main.item(), batch_size // 2)
            t_top1_aux.update(t_prec1_aux.item(), batch_size // 2)



        # compute gradient and do SGD step
        optimizer.zero_grad()
        s_loss.backward(retain_graph=True)
        t_loss.backward(retain_graph=True)
        s_cps_loss.backward()
        t_cps_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if not mixbn:
            s_loss_str = "{:.4f}".format(s_losses.avg)
            t_loss_str = "{:.4f}".format(t_losses.avg)
            s_top1_str = "{:.4f}".format(s_top1.avg)
            t_top1_str = "{:.4f}".format(t_top1.avg)

        else:
            # s_loss_str = "{:.2f}/{:.2f}/{:.2f}".format(s_losses.avg, s_losses_main.avg, s_losses_aux.avg)
            # t_loss_str = "{:.2f}/{:.2f}/{:.2f}".format(t_losses.avg, t_losses_main.avg, t_losses_aux.avg)
            s_loss_str = "{:.2f}".format(s_losses.avg)
            t_loss_str = "{:.2f}".format(t_losses.avg)
            s_cps_loss_str = "{:.2f}".format(s_cps_losses.avg)
            t_cps_loss_str = "{:.2f}".format(t_cps_losses.avg)
            # s_top1_str = "{:.2f}/{:.2f}/{:.2f}".format(s_top1.avg, s_top1_main.avg, s_top1_aux.avg)
            # t_top1_str = "{:.2f}/{:.2f}/{:.2f}".format(t_top1.avg, t_top1_main.avg, t_top1_aux.avg)
            s_top1_str = "{:.2f}".format(s_top1.avg)
            t_top1_str = "{:.2f}".format(t_top1.avg)

        bar.suffix = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.2f}s | Total: {total:} | ETA: {eta:} | Loss_s: {s_loss:s} | Loss_t: {t_loss:s} | Loss_cps_s: {s_cps_loss:s} | Loss_cps_t: {t_cps_loss:s} | s_top1: {s_top1:s} | t_top1: {t_top1:s} | s_top5: {s_top5: .1f} | t_top5: {t_top5: .1f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            s_loss=s_loss_str,
            t_loss=t_loss_str,
            s_cps_loss=s_cps_loss_str,
            t_cps_loss=t_cps_loss_str,
            s_top1=s_top1_str,
            t_top1=t_top1_str,
            s_top5=s_top5.avg,
            t_top5=t_top5.avg,
        )
        bar.next()
    bar.finish()
    if mixbn:
        result = ((s_losses.avg, s_cps_losses.avg, s_top1.avg, s_losses_main.avg, s_losses_aux.avg, s_top1_main.avg, s_top1_aux.avg),
                 (t_losses.avg, t_cps_losses.avg, t_top1.avg, t_losses_main.avg, t_losses_aux.avg, t_top1_main.avg, t_top1_aux.avg))
        return result
    else:
        result = (s_losses.avg, s_top1.avg), (t_losses.avg, t_top1.avg)
        return result


def test(val_loader, model, step, criterion, epoch, use_cuda, FGSM=False):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        if FGSM:
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(shape=(3, 1, 1)).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(shape=(3, 1, 1)).cuda()
            inputs.requires_grad = True
            outputs = model(inputs, step=step)
            loss = torch.nn.functional.nll_loss(outputs, targets)
            model.zero_grad()
            loss.backward()
            data_grad = inputs.grad.data
            epsilon = 1 / std / 255.0 * 16.0
            upper_bound = (1 - mean) / std
            lower_bound = (0 - mean) / std
            inputs = fgsm_attack(inputs, epsilon, data_grad, upper_bound, lower_bound)

        # compute output
        with torch.no_grad():
            outputs = model(inputs, step=step)
            if args.imagenet_a:
                outputs = outputs[:, indices_in_1k]
            loss = criterion(outputs, targets).mean()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def show_performance(distortion_name, model, criterion, start_epoch, use_cuda):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    errs = []

    for severity in range(1, 6):
        valdir = os.path.join(args.data, distortion_name, str(severity))
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                # transforms.Scale(256),
                # transforms.CenterCrop(224), # already 224 x 224
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)

        errs.append(1. - test_acc / 100.)

    print('\n=Average', tuple(errs))
    return np.mean(errs)


def save_checkpoint(state, is_best, is_shape, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        if is_shape:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'shape_model_best.pth.tar'))
        else:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'texture_model_best.pth.tar'))


if __name__ == '__main__':
    main()
