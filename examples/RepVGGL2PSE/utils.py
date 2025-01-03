# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# The training script is based on the code of Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# --------------------------------------------------------

import torch
import math
import os
from collections import OrderedDict

Clp_max = [2, 2, 2]
quantized_weight = 32

def quantize_to_bit_(x, nbit):
    x = (1-2.0**(1-nbit))*x
    x = torch.clamp(x,-1,1)
    x = torch.round(torch.div(x, 2.0**(1-nbit)))
    return x

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    ckpt = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            ckpt[k[7:]] = v
        else:
            ckpt[k] = v
    model.load_state_dict(ckpt)


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, T_cosine_max, eta_min=0, last_epoch=-1, warmup=0):
        self.eta_min = eta_min
        self.T_cosine_max = T_cosine_max
        self.warmup = warmup
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            return [self.last_epoch / self.warmup * base_lr for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup) / (self.T_cosine_max - self.warmup))) / 2
                    for base_lr in self.base_lrs]


def log_msg(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        print(message, file=f)



try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present."""
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


def load_checkpoint(config, model, optimizer, lr_scheduler, logger, model_ema=None):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    ckpt = checkpoint['model']
    keys_to_modify = list(ckpt.keys())
    # for k in keys_to_modify:
    #     # print(k)
    #     if 'linear' in k or 'aux.5' in k:
    #         del ckpt[k]

    msg = model.load_state_dict(ckpt, strict=False)
    # msg = model.load_state_dict(checkpoint, strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        # if 'max_accuracy' in checkpoint:
            # max_accuracy = checkpoint['max_accuracy']
    if model_ema is not None:
        unwrap_model(model_ema).load_state_dict(checkpoint['ema'])
        print('=================================================== EMAloaded')

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_weights(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    unwrap_model(model).load_state_dict(checkpoint, strict=False)
    print('=================== loaded from', path)

def load_snn_weights(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    dst_dict = model.state_dict()
    src_dict = checkpoint
    f = open('dstname.txt','w')
    for i in dst_dict.keys():
        f.write(str(i)+'\n')
    f.close()
    f = open('srcname.txt','w')
    for i in src_dict.keys():
        f.write(str(i)+'\n')
    f.close()
    reshape_dict = {}
    for (k, v) in src_dict.items():
        if k in dst_dict.keys():
            dst_dict[k] = torch.tensor(dst_dict[k], dtype=torch.float32)
            v = torch.tensor(v, dtype=torch.float32)
            # if 'weight' in k and ('stage1' in k or 'stage4' in k):
            #     reshape_dict[k] = torch.nn.Parameter(v.reshape(dst_dict[k].shape) * Clp_max[2])
            # if 'weight' in k and ('stage4' in k or 'stage3_second' in k):
            #     reshape_dict[k] = torch.nn.Parameter(v.reshape(dst_dict[k].shape) * Clp_max[1])
            if 'weight' in k:
                reshape_dict[k] = torch.nn.Parameter(v.reshape(dst_dict[k].shape) * 2)
            # elif 'weight' in k:
            #     reshape_dict[k] = torch.nn.Parameter(v.reshape(dst_dict[k].shape) * 2)
            else:
                reshape_dict[k] = torch.nn.Parameter(v.reshape(dst_dict[k].shape))
            reshape_dict['gap.weight'] = dst_dict['gap.weight'] * 2
            # reshape_dict['avgpool.weight'] = dst_dict['avgpool.weight'] *Clp_max
    # theta = []
    # for (k, v) in reshape_dict.items():
    #     if 'rbr_reparam' in k and 'weight' in k:
    #         k_bias = k[:-6]
    #         k_bias += "bias"
    #         v = torch.tensor(v, dtype=torch.float32)
    #         reshape_dict[k_bias] = torch.tensor(reshape_dict[k_bias], dtype=torch.float32)
    #         factor = torch.max(torch.abs(v))
    #         if factor<torch.max(torch.abs(reshape_dict[k_bias])):
    #             factor = torch.max(torch.abs(reshape_dict[k_bias]))
    #         v /= factor
    #         v = torch.nn.Parameter(quantize_to_bit_(v, quantized_weight))
    #         reshape_dict[k_bias]/=factor
    #         reshape_dict[k_bias]= torch.nn.Parameter(quantize_to_bit_(reshape_dict[k_bias], quantized_weight))
    #         layer_theta = 2.0**(quantized_weight-1)*(1-2.0**(1-quantized_weight))*2/factor
    #         theta.append(layer_theta)

    unwrap_model(model).load_state_dict(reshape_dict, strict=False)
    print('=================== loaded from', path)

def save_latest(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, model_ema=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()
    if model_ema is not None:
        save_state['ema'] = unwrap_model(model_ema).state_dict()

    save_path = os.path.join(config.OUTPUT, 'latest.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, is_best=False, model_ema=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()
    if model_ema is not None:
        save_state['ema'] = unwrap_model(model_ema).state_dict()

    if is_best:
        best_path = os.path.join(config.OUTPUT, 'best_ckpt.pth')
        torch.save(save_state, best_path)

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


import torch.distributed as dist

def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth') and 'ema' not in ckpt]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def update_model_ema(cfg, num_gpus, model, model_ema, cur_epoch, cur_iter):
    """Update exponential moving average (ema) of model weights."""
    update_period = cfg.TRAIN.EMA_UPDATE_PERIOD
    if update_period is None or update_period == 0 or cur_iter % update_period != 0:
        return
    # Adjust alpha to be fairly independent of other parameters
    total_batch_size = num_gpus * cfg.DATA.BATCH_SIZE
    adjust = total_batch_size / cfg.TRAIN.EPOCHS * update_period
    # print('ema adjust', adjust)
    alpha = min(1.0, cfg.TRAIN.EMA_ALPHA * adjust)
    # During warmup simply copy over weights instead of using ema
    alpha = 1.0 if cur_epoch < cfg.TRAIN.WARMUP_EPOCHS else alpha
    # Take ema of all parameters (not just named parameters)
    params = unwrap_model(model).state_dict()
    for name, param in unwrap_model(model_ema).state_dict().items():
        param.copy_(param * (1.0 - alpha) + params[name] * alpha)