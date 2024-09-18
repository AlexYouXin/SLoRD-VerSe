import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, print_network
# from torchvision import transforms
from torch.cuda.amp import autocast as autocast
import math
from utils import contour_loss_module


def train(model, writer, optimizer, dice_loss, ce_loss, center_loss, contour_loss, args, epoch_num, trainloader, snapshot_path, lamb1, lamb2, V):

    max_iterations = args.max_epochs * len(trainloader)
    max_epoch = args.max_epochs
    iter_num = epoch_num * len(trainloader)
    model.train()

    best_performance = 0.0

    mean_loss = 0
    mean_dice = 0
    mean_loss_ce = 0
    mean_loss_center = 0
    mean_loss_contour = 0
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch, center_batch, contour_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['center'], sampled_batch['contour']
        image_batch, label_batch, center_batch, contour_batch, V = image_batch.cuda(), label_batch.cuda(), center_batch.cuda(), contour_batch.cuda(), V.cuda()

        outputs, center, coefficient = model(image_batch)

        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch[:].long(), softmax=True)

        loss_center = center_loss(center, center_batch)
        # loss_contour = 0
        # if epoch_num > 100:
        loss_contour = 0
        if epoch_num > 100:
            loss_contour = contour_loss(coefficient, contour_batch, V, center)

        loss = 0.5 * loss_ce + 0.5 * loss_dice + lamb1 * loss_center + lamb2 * loss_contour
        dice = 1 - loss_dice
        mean_loss += loss
        mean_dice += dice
        mean_loss_ce += loss_ce
        mean_loss_center += loss_center
        mean_loss_contour += loss_contour
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            lr_ = param_group['lr']
        iter_num = iter_num + 1
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/train_total_loss', loss, iter_num)
        writer.add_scalar('info/train_loss_ce', loss_ce, iter_num)
        writer.add_scalar('info/train_dice', dice, iter_num)

        if iter_num % 35 == 0:
            logging.info('epoch : %d, iteration : %d, train loss : %f, train loss_ce: %f, train loss_dice: %f, train dice : %f, train loss center: %f, train loss contour: %f' % (
                epoch_num, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item(), loss_center.item(), loss_contour))

        
    # mean loss and mean dice
    mean_loss = float(mean_loss / len(trainloader))
    mean_dice = float(mean_dice / len(trainloader))
    mean_loss_ce = float(mean_loss_ce / len(trainloader))
    mean_loss_center = float(mean_loss_center / len(trainloader))
    mean_loss_contour = float(mean_loss_contour / len(trainloader))
    writer.add_scalar('info/epoch_train_total_loss', mean_loss, epoch_num)
    writer.add_scalar('info/epoch_train_dice', mean_dice, epoch_num)
    writer.add_scalar('info/epoch_train_total_loss_ce', mean_loss_ce, epoch_num)
    # print('epoch :', epoch_num, 'Train Loss :', mean_loss, 'Train dice :', mean_dice)
    logging.info('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f, mean train loss center: %f, mean train loss contour: %f' % (epoch_num, mean_loss, mean_loss_ce, mean_dice, mean_loss_center, mean_loss_contour))

    save_interval = 25  # int(max_epoch/5)
    # if epoch_num > int(max_epoch / 5) and (epoch_num + 1) % save_interval == 0:
    if (epoch_num + 1) >= 50 and (epoch_num + 1) % save_interval == 0:
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num + 1) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

    return writer, mean_dice


def validation(model, writer, dice_loss, ce_loss, center_loss, contour_loss, args, epoch_num, valloader, snapshot_path, lamb1, lamb2, V):
    model.eval()
    iter_num = epoch_num * len(valloader)

    mean_loss = 0
    mean_dice = 0
    mean_loss_ce = 0
    mean_loss_center = 0
    mean_loss_contour = 0
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valloader):
            image_batch, label_batch, center_batch, contour_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['center'], sampled_batch['contour']
            image_batch, label_batch, center_batch, contour_batch, V = image_batch.cuda(), label_batch.cuda(), center_batch.cuda(), contour_batch.cuda(), V.cuda()
            # contour_batch = contour_batch / 25
            outputs, center, coefficient = model(image_batch)
            # print(outputs.shape, label_batch.shape)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            loss_center = center_loss(center, center_batch)         # scale_factor: 800, 取前512维
            loss_contour = 0
            if epoch_num > 100:
                loss_contour = contour_loss(coefficient, contour_batch, V, center)             # scale_factor: (32, 56, 48)
    
            loss = 0.5 * loss_ce + 0.5 * loss_dice + lamb1 * loss_center + lamb2 * loss_contour


            dice = 1 - loss_dice

            mean_loss += loss
            mean_dice += dice
            mean_loss_ce += loss_ce
            mean_loss_center += loss_center
            mean_loss_contour += loss_contour
            writer.add_scalar('info/val_total_loss', loss, iter_num)
            writer.add_scalar('info/val_loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/val_dice', dice, iter_num)


            iter_num += 1

            if iter_num % 10 == 0:
                logging.info('epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, val loss_dice: %f, val dice : %f, val loss center: %f, val loss contour: %f' % (
                    epoch_num, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item(), loss_center.item(), loss_contour))



    # mean loss and mean dice
    mean_loss = float(mean_loss / len(valloader))
    mean_dice = float(mean_dice / len(valloader))
    mean_loss_ce = float(mean_loss_ce / len(valloader))
    mean_loss_center = float(mean_loss_center / len(valloader))
    mean_loss_contour = float(mean_loss_contour / len(valloader))
    writer.add_scalar('info/epoch_val_total_loss', mean_loss, epoch_num)
    writer.add_scalar('info/epoch_val_dice', mean_dice, epoch_num)
    writer.add_scalar('info/epoch_val_total_loss_ce', mean_loss_ce, epoch_num)
    writer.add_scalar('info/epoch_val_loss_center', mean_loss_center, epoch_num)
    writer.add_scalar('info/epoch_val_loss_coefficient', mean_loss_contour, epoch_num)
    # print('epoch :', epoch_num, 'validation Loss :', mean_loss, 'validation dice :', mean_dice)
    logging.info('epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f, mean val loss center: %f, mean val loss contour: %f' % (epoch_num, mean_loss, mean_loss_ce, mean_dice, mean_loss_center, mean_loss_contour))

    return writer, mean_dice


def run_main(args, model, snapshot_path):
    from datasets.dataset import verse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    batch_size = args.batch_size * args.n_gpu
    # random crop 224 * 224 patches from 512 * 512 patches
    db_train = verse_dataset(base_dir=args.train_root_path, list_dir=args.list_dir, split="train", num_classes=args.num_classes, args=args)

    print("The length of train set is: {}".format(len(db_train)))
    # random crop 224 * 224 patches from 512 * 512 patches
    db_val = verse_dataset(base_dir=args.val_root_path, list_dir=args.list_dir, split="val", num_classes=args.num_classes, args=args)
    
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,      # m_worker: 8    ---->>   4
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,      # m_worker: 8    ---->>   4
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    parameter_num, net = print_network(model)
    logging.info("Total number of network parameters: {}".format(parameter_num))
    logging.info("network structure: {}".format(net))

    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)          
    # max_iterations = args.max_iterations

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    writer = SummaryWriter(snapshot_path + '/log')
    # load progress bar
    iterator = tqdm(range(max_epoch), ncols=70)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-5)           # 5e-4
    
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 44, eta_min=0, last_epoch=-1, verbose=False)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 25, eta_min=5e-6)
    #  CosineAnnealingWarmRestarts
    # record best model with highest train and val dice
    highest_train_dice = 0
    epo_train = 1
    highest_val_dice = 0
    epo_val = 1
    # these two weight need to be changed
    dice_weight = [0.2] + [0.8]
    ce_weight = [0.2] + [0.8]                # [1.0, 1.0]
    lamb1, lamb2 = 1, 1
    class_weights = torch.FloatTensor(ce_weight).cuda()
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(args.num_classes, dice_weight)
    
    center_loss = nn.MSELoss()
    contour_loss_ = contour_loss_module(num_contour=3, base_num=args.coefficient, args=args)

    mat = np.load('xxxxx/verse19_distance_resample-5.npy')
    scale = [20, 20, 20]


    max_dist = math.sqrt(scale[0] / 2 * scale[0] / 2 + scale[1] / 2 * scale[1] / 2 + scale[2] / 2 * scale[2] / 2)
    mat = mat / max_dist
    n, r, c = mat.shape
    mat = np.reshape(mat, (n, -1))
    U, S, V = np.linalg.svd(mat)
    V = torch.from_numpy(V.astype(np.float32)).float()
    # print('V: ', V.shape)
    ### train and validation
    for epoch_num in iterator:
        # if epoch_num < 100:
            # lamb2 = 0
        
        writer, train_dice = train(model, writer, optimizer, dice_loss, ce_loss, center_loss, contour_loss_, args, epoch_num, trainloader, snapshot_path, lamb1, lamb2, V)
        writer, val_dice = validation(model, writer, dice_loss, ce_loss, center_loss, contour_loss_, args, epoch_num, valloader, snapshot_path, lamb1, lamb2, V)
        if train_dice > highest_train_dice:
            highest_train_dice = train_dice
            epo_train = epoch_num + 1
        if val_dice > highest_val_dice:
            highest_val_dice = val_dice
            epo_val = epoch_num + 1
            save_mode_path = os.path.join(snapshot_path, 'best_model' + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save best model to {} at epoch {}, val dice: {}".format(save_mode_path, epo_val, val_dice))

        lr_scheduler.step()
        torch.cuda.empty_cache()

    logging.info('highest train dice: %f at epoch %d, highest val dice : %f at epoch %d' % (
        highest_train_dice, epo_train, highest_val_dice, epo_val))


    writer.close()
    return "Training and validation Finished!"