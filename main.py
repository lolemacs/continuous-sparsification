import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import random
from models import *
from load_datasets import *

parser = argparse.ArgumentParser(description='Training a ResNet on CIFAR-10 with Continuous Sparsification')
parser.add_argument('--which-gpu', type=int, default=0, help='which GPU to use')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training/val/test (default: 128)')
parser.add_argument('--epochs', type=int, default=85, help='number of epochs to train (default: 85)')
parser.add_argument('--rounds', type=int, default=3, help='number of rounds to train (default: 3)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1234)')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--val-set-size', type=int, default=5000, help='how much of the training set to use for validation (default: 5000)')
parser.add_argument('--lr-schedule', type=int, nargs='+', default=[56,71], help='epochs at which the learning rate will be dropped')
parser.add_argument('--lr-drops', type=float, nargs='+', default=[0.1, 0.1], help='how much to drop the lr at each epoch in the schedule')
parser.add_argument('--decay', type=float, default=0.0001, help='weight decay (default: 0.0001)')
parser.add_argument('--rewind-epoch', type=int, default=2, help='epoch to rewind weights to (default: 2)')
parser.add_argument('--lmbda', type=float, default=1e-8, help='lambda for L1 mask regularization (default: 1e-8)')
parser.add_argument('--final-temp', type=float, default=200, help='temperature at the end of each round (default: 200)')
parser.add_argument('--mask-initial-value', type=float, default=0., help='initial value for mask parameters')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.which_gpu)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

train_loader, val_loader, test_loader = generate_loaders(args.val_set_size, args.batch_size, args.workers)

model = ResNet(args.mask_initial_value)

if args.cuda: model.cuda()
print(model)

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    assert len(args.lr_schedule) == len(args.lr_drops), "length of gammas and schedule should be equal"
    for (drop, step) in zip(args.lr_drops, args.lr_schedule):
        if (epoch >= step): lr = lr * drop
        else: break
    for param_group in optimizer.param_groups: param_group['lr'] = lr

def compute_remaining_weights(masks):
    return 1 - sum(float((m == 0).sum()) for m in masks) / sum(m.numel() for m in masks)

def train(outer_round):
    for epoch in range(args.epochs):
        print('\t--------- Epoch {} -----------'.format(epoch))
        model.train()
        if epoch > 0: model.temp *= temp_increase  
        if outer_round == 0 and epoch == args.rewind_epoch: model.checkpoint()
        for optimizer in optimizers: adjust_learning_rate(optimizer, epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda: data, target = data.cuda(), target.cuda(non_blocking=True)
            for optimizer in optimizers: optimizer.zero_grad()
            output = model(data)
            pred = output.max(1)[1]
            batch_correct = pred.eq(target.data.view_as(pred)).sum()
            masks = [m.mask for m in model.mask_modules]
            entries_sum = sum(m.sum() for m in masks)
            loss = F.cross_entropy(output, target) + args.lmbda * entries_sum
            loss.backward()
            for optimizer in optimizers: optimizer.step()

        val_acc = test(val_loader)
        test_acc = test(test_loader)
        remaining_weights = compute_remaining_weights(masks)
        print('\t\tTemp: {:.1f}\tRemaining weights: {:.4f}\tVal acc: {:.1f}\tTest acc: {}'.format(model.temp, remaining_weights, val_acc, test_acc))
        
def test(loader):
    model.eval()
    correct = 0.
    total = 0.
    with torch.no_grad():
        for data, target in loader:
            if args.cuda: data, target = data.cuda(), target.cuda(non_blocking=True)
            output = model(data)
            pred = output.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            total += data.size()[0]
    acc = 100. * correct.item() / total
    return acc

iters_per_reset = args.epochs-1
temp_increase = args.final_temp**(1./iters_per_reset)

trainable_params = filter(lambda p: p.requires_grad, model.parameters())
num_params = sum([p.numel() for p in trainable_params])
print("Total number of parameters: {}".format(num_params))

weight_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask' not in p[0], model.named_parameters()))
mask_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask' in p[0], model.named_parameters()))

model.ticket = False
weight_optim = optim.SGD(weight_params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.decay)
mask_optim = optim.SGD(mask_params, lr=args.lr, momentum=0.9, nesterov=True)
optimizers = [weight_optim, mask_optim]
for outer_round in range(args.rounds):
    print('--------- Round {} -----------'.format(outer_round))
    train(outer_round)
    model.temp = 1
    if outer_round != args.rounds-1: model.prune()

print('--------- Training final ticket -----------')
optimizers = [optim.SGD(weight_params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.decay)]
model.ticket = True
model.rewind_weights()
train(outer_round)
