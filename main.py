import os
import time

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

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
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    raw_model = sfocus18(num_classes, args.parallel_last_layers)
    model = nn.DataParallel(raw_model, device_ids=[idx for idx in range(args.num_gpus)]).cuda()
    wandb.watch(model, log_freq=100)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.lr_decay)
    cudnn.benchmark = True

    dirname = 'trained_models'
    fname = time.strftime('%y%m%d_%H%M%S.pt')
    path = os.path.join(dirname, fname)
    os.makedirs(dirname, exist_ok=True)

    best_top1_acc = 0
    best_top5_acc = 0

    for epoch in range(args.start_epoch, args.num_epochs):

        train(train_dl, model, criterion, optimizer, epoch, classification_only=args.classification_only)
        top1_acc, top5_acc = validate(val_dl, model, criterion)
        scheduler.step()

        wandb.log({
            'Acc/ValidationTop1': top1_acc,
            'Acc/ValidationTop5': top5_acc,
        })

        is_top1_best = top1_acc > best_top1_acc
        if is_top1_best: best_top1_acc = top1_acc

        is_top5_best = top5_acc > best_top5_acc
        if is_top5_best: best_top5_acc = top5_acc

        if is_top1_best and is_top5_best:
            save_training(model, optimizer, scheduler, best_top1_acc, best_top5_acc, path)

def train(
    train_dl,
    model,
    criterion,
    optimizer,
    epoch,
    batch_print_frequency=100,
    classification_only=False,
):

    batch_size = train_dl.batch_size

    batch_time_meter = AverageMeter()
    loss_meter = AverageMeter()
    top1_acc_meter = AverageMeter()
    top5_acc_meter = AverageMeter()

    model.train()

    for idx, (inputs, labels) in enumerate(train_dl):

        start_time = time.time()

        inputs = inputs.cuda()
        labels = labels.cuda()

        if classification_only:
            outputs = model(inputs, labels, classification_only)
            ce_loss = criterion(outputs, labels)
            loss = ce_loss

            ac_loss, as_in_loss, as_la_loss, bw_loss = (0, 0, 0, 0)
        else:
            if model.module.parallel_last_layers:
                outputs, A_true_la, A_conf_la, ac_loss, as_in_loss, as_la_loss, bw_loss = model(inputs, labels)
                ce_loss = criterion(outputs, labels)
                loss = ce_loss + ac_loss + as_in_loss + as_la_loss + bw_loss
            else:
                outputs, A_true_la, A_conf_la, ac_loss, as_in_loss, as_la_loss = model(inputs, labels)
                ce_loss = criterion(outputs, labels)
                loss = ce_loss + ac_loss + as_in_loss + as_la_loss

                bw_loss = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        top1_acc, top5_acc = check_accuracy(outputs, labels, topk=(1, 5))

        batch_time = time.time() - start_time

        wandb.log({
            'Epoch': epoch,
            'BatchTime': batch_time,
            'Loss/CrossEntropy': ce_loss,
            'Loss/AttConsistency': ac_loss,
            'Loss/AttSepInner': as_in_loss,
            'Loss/AttSepLast': as_la_loss,
            'Loss/BlackWhite': bw_loss,
            'Loss/Total': loss,
            'Acc/Top1': top1_acc,
            'Acc/Top5': top5_acc,
        })

        batch_time_meter.update(batch_time)
        loss_meter.update(loss.item(), batch_size)
        top1_acc_meter.update(top1_acc, batch_size)
        top5_acc_meter.update(top5_acc, batch_size)

        if idx % batch_print_frequency == 0:
            print(
                f'Epoch [{epoch}][{idx}/{len(train_dl)}]\n'
                f'Time  {batch_time_meter.val:.4f} ({batch_time_meter.avg:.4f})\n'
                f'Loss  {loss_meter.val:.4f} ({loss_meter.avg:.4f})\n'
                f'Acc@1 {top1_acc_meter.val:.4f} ({top1_acc_meter.avg:.4f})\n'
                f'Acc@5 {top5_acc_meter.val:.4f} ({top5_acc_meter.avg:.4f})'
            )

def validate(val_dl, model, criterion):

    batch_size = val_dl.batch_size

    top1_acc_meter = AverageMeter()
    top5_acc_meter = AverageMeter()

    model.eval()

    for idx, (inputs, labels) in enumerate(val_dl):

        inputs = inputs.cuda()
        labels = labels.cuda()

        if model.module.parallel_last_layers:
            outputs, A_true_la, A_conf_la, ac_loss, as_in_loss, as_la_loss, bw_loss = model(inputs, labels)
            loss = criterion(outputs, labels) + ac_loss + as_in_loss + as_la_loss + bw_loss
        else:
            outputs, A_true_la, A_conf_la, ac_loss, as_in_loss, as_la_loss = model(inputs, labels)
            loss = criterion(outputs, labels) + ac_loss + as_in_loss + as_la_loss

        top1_acc, top5_acc = check_accuracy(outputs, labels, topk=(1, 5))

        top1_acc_meter.update(top1_acc, batch_size)
        top5_acc_meter.update(top5_acc, batch_size)

    return top1_acc_meter.avg, top5_acc_meter.avg

def check_accuracy(outputs, labels, topk=(1, 5)):

    maxk = max(topk)
    batch_size = len(labels)

    _, topk_indices = torch.topk(outputs, maxk, dim=1)
    expanded_labels = labels.unsqueeze(1).expand_as(topk_indices)
    correct_idx = (topk_indices == expanded_labels)

    acc = []
    for k in topk:
        topk_acc = torch.sum(correct_idx[:,:k]) / batch_size
        acc.append(topk_acc)

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


if __name__ == '__main__':

    def test_sanity_accuracy():

        from icecream import ic

        batch_size = 4
        num_classes = 5

        outputs = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        ic(outputs, labels)

        topk = (1, 3)
        maxk = max(topk)
        val, idx = torch.topk(outputs, maxk, dim=1)
        ic(idx)
        expanded_labels = labels.unsqueeze(1).expand_as(idx)
        ic(expanded_labels)
        correct_idx = idx == expanded_labels
        ic(correct_idx)

        for k in topk:

            ic(torch.sum(correct_idx[:,:k]) / batch_size)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--dataset-name', type=str, default='TinyImageNet')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--classification-only', type=bool, default=False)
    parser.add_argument('--parallel-last-layers', type=bool, default=True)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-decay', type=float, default=0.9)
    parser.add_argument('--milestones', type=int, default=[50,100], nargs='+')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)

    parser.add_argument('--sigma-weight', type=float, default=0.55)
    parser.add_argument('--omega', type=int, default=100)
    parser.add_argument('--theta', type=float, default=0.8)

    parser.add_argument('--wandb-key', type=str, default='0f5cd9050587f427bc738060f38f870174f2c8e4')
    parser.add_argument('--wandb-user', type=str, default='hphp')
    parser.add_argument('--wandb-project', type=str, default='ICASC++')
    parser.add_argument('--wandb-mode', type=str, default='online')
    parser.add_argument('--exp-name', type=str, default='TinyImageNet')

    args = parser.parse_args()

    main(args)