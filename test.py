import os
import time
import gc

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib as mpl

from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from utils.data_utils import get_datasets
from utils.average_meter import AverageMeter
from models.sharpen_focus import sfocus18


def main(args):

    os.environ['WANDB_API_KEY'] = args.wandb_key
    os.environ['WANDB_MODE'] = args.wandb_mode
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_user,
        name=args.exp_name,
        reinit=True
    )
    wandb.config.update(args)

    train_dataset, val_dataset, num_classes, unorm = get_datasets(args.dataset_name)
    val_dl = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    raw_model = sfocus18(num_classes, parallel_block_channels=args.parallel_block_channels)
    model = nn.DataParallel(raw_model, device_ids=[idx for idx in range(args.num_gpus)]).cuda()
    load_training(args.model_path, model)

    num_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f"Number of model parameters: {num_parameters}")
    wandb.config.update({'num_parameters': num_parameters})
    cudnn.benchmark = True

    # TODO
    # Add one extra output to validate function (H-score)
    top1_acc, top5_acc, true_overlays, conf_overlays = validate(val_dl, model)
    # top1_acc, top5_acc, h_score, true_overlays, conf_overlays = validate(val_dl, model)
    print(top1_acc, top5_acc)

    wandb.log({
        'AttentionMap/True': true_overlays,
        'AttentionMap/Conf': conf_overlays,
        'Acc/ValidationTop1': top1_acc,
        'Acc/ValidationTop5': top5_acc,
        # 'Add/HScore': h_score,
    })

    del true_overlays, conf_overlays
    gc.collect()
    torch.cuda.empty_cache()

def validate(val_dl, model):

    batch_size = val_dl.batch_size

    top1_acc_meter = AverageMeter()
    top5_acc_meter = AverageMeter()

    model.eval()

    for i, (inputs, labels) in enumerate(val_dl):

        inputs = inputs.cuda()
        labels = labels.cuda()

        if model.module.parallel_block_channels:
            outputs, A_true_la, A_conf_la, ac_loss, as_in_loss, as_la_loss, bw_loss = model(inputs, labels)
        else:
            outputs, A_true_la, A_conf_la, ac_loss, as_in_loss, as_la_loss = model(inputs, labels)

        top1_acc, top5_acc = check_accuracy(outputs, labels, topk=(1, 5))

        top1_acc_meter.update(top1_acc, batch_size)
        top5_acc_meter.update(top5_acc, batch_size)

        with torch.no_grad():

            A_true_la = A_true_la.cpu()
            A_conf_la = A_conf_la.cpu()

            if i == 0:

                alpha = 0.6
                cmap = mpl.colormaps.get_cmap('coolwarm')

                true_overlays = []
                conf_overlays = []

                anno_file = open('dataset/tiny-imagenet-200/val/val_annotations.txt', 'r')
                word_file = open('dataset/tiny-imagenet-200/words.txt')
                ids = []; categories = []
                for x in word_file.readlines():
                    id, category = x.split('\t')
                    ids.append(id); categories.append(category)

                for j in range(batch_size // 2):
                    coords = []
                    line = anno_file.readline().split('\t')
                    for k, word in enumerate(line):
                        if k == 1:
                            for l, id in enumerate(ids):
                                if word == id:
                                    category = categories[l]
                        elif k >= 2:
                            coords.append(int(word))

                    image = inputs[j]
                    image -= image.amin(dim=(1, 2), keepdim=True)
                    image /= image.amax(dim=(1, 2), keepdim=True) + 1e-6

                    true_mask = F.interpolate(A_true_la[j].unsqueeze(0), size=image.shape[-2:], mode='bilinear').squeeze()
                    true_mask -= true_mask.min()
                    true_mask /= true_mask.max() + 1e-6
                    true_hmap = (cmap(true_mask)[:,:,:3] * 255).astype(np.uint8)

                    conf_mask = F.interpolate(A_conf_la[j].unsqueeze(0), size=image.shape[-2:], mode='bilinear').squeeze()
                    conf_mask -= conf_mask.min()
                    conf_mask /= conf_mask.max() + 1e-6
                    conf_hmap = (cmap(conf_mask)[:,:,:3] * 255).astype(np.uint8)

                    image = to_pil_image(image)
                    true_hmap = Image.fromarray(true_hmap)
                    conf_hmap = Image.fromarray(conf_hmap)

                    true_overlay = Image.blend(image, true_hmap, alpha)
                    draw = ImageDraw.Draw(true_overlay)
                    draw.rectangle(coords, outline=(0, 255, 0))

                    conf_overlay = Image.blend(image, conf_hmap, alpha)
                    draw = ImageDraw.Draw(conf_overlay)
                    draw.rectangle(coords, outline=(0, 255, 0))

                    true_overlays.append(wandb.Image(true_overlay, caption=category))
                    conf_overlays.append(wandb.Image(conf_overlay, caption=category))

                    # plt.figure()
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(true_overlay)
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(conf_overlay)
                    # plt.show()

    return top1_acc_meter.avg, top5_acc_meter.avg, true_overlays, conf_overlays

def check_accuracy(outputs, labels, topk=(1, 5)):

    maxk = max(topk)
    batch_size = len(labels)

    _, topk_indices = torch.topk(outputs, maxk, dim=1)
    expanded_labels = labels.unsqueeze(1).expand_as(topk_indices)
    correct_idx = (topk_indices == expanded_labels)

    acc = []
    for k in topk:
        topk_acc = torch.sum(correct_idx[:,:k]) / batch_size
        acc.append(topk_acc.item())

    return acc

def save_training(model, optimizer, scheduler, best_top1_acc, best_top5_acc, path):

    state_dict = {
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_top1_acc': best_top1_acc,
        'best_top5_acc': best_top5_acc,
    }

    torch.save(state_dict, path)

def load_training(path, model):

    state_dict = torch.load(path)

    model.load_state_dict(state_dict['model_state_dict'])


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--dataset-name', type=str, default='TinyImageNet')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=64)

    # if 0, no parallel blocks
    # else, # of channels of the parallel blocks are set accordingly
    parser.add_argument('--parallel-block-channels', type=int, default=0)

    parser.add_argument('--wandb-key', type=str, default='0f5cd9050587f427bc738060f38f870174f2c8e4')
    parser.add_argument('--wandb-user', type=str, default='hphp')
    parser.add_argument('--wandb-project', type=str, default='ICASC++')
    parser.add_argument('--wandb-mode', type=str, default='online')
    parser.add_argument('--exp-name', type=str, default='TinyImageNet')

    parser.add_argument('--model-path', type=str, default='')

    args = parser.parse_args()

    main(args)
