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
from torchvision import transforms


def train(model, writer, optimizer, dice_weight, ce_weight, args, epoch_num, trainloader, snapshot_path):
    # base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_epochs * len(trainloader)
    max_epoch = args.max_epochs
    iter_num = epoch_num * len(trainloader)
    model.train()
    class_weights = torch.FloatTensor(ce_weight).cuda()
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(num_classes, dice_weight)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001, nesterov=True)

    best_performance = 0.0

    mean_loss = 0
    mean_dice = 0
    mean_loss_ce = 0
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        outputs = model(image_batch)
        # print('input, predicted output, label size:', image_batch.shape, outputs.shape, label_batch.shape)    # ([1, 1, 128, 128, 128]), ([1, 26, 128, 128, 128]), ([1, 128, 128, 128])
        '''
        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        dice = 1 - loss_dice
        '''
        loss_ce_1 = ce_loss(outputs, label_batch[:].long())
        loss_dice_1 = dice_loss(outputs, label_batch[:].long(), softmax=True)


        loss_ce = loss_ce_1
        loss_dice = loss_dice_1

        loss = 0.5 * loss_ce + 0.5 * loss_dice
        dice = 1 - loss_dice_1           # dice = 1 - loss_dice
        mean_loss += loss
        mean_dice += dice
        mean_loss_ce += loss_ce
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

        if iter_num % 8 == 0:
            logging.info('epoch : %d, iteration : %d, train loss : %f, train loss_ce: %f, train loss_dice: %f, train dice : %f' % (
                epoch_num, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))

        '''
        if epoch_num > 800 and epoch_num % 50 == 0:
            # visualize
            image_numpy = image_batch[0].cpu().detach().numpy()
            target_numpy = label_batch[0].unsqueeze(0).cpu().detach().numpy()
            output_numpy = outputs[0].unsqueeze(0).cpu().detach().numpy()
            output_numpy = np.argmax(output_numpy, axis=1)
            npy_file = np.concatenate((image_numpy, output_numpy, target_numpy), axis=0)
            np.save('/lustre/home/acct-eeyj/eeyj-wr/youxin/vertebrae_more_data/trans3dunet/predict_gt/train/epoch {}/idx {}.npy'.format(epoch_num, i_batch), npy_file)
        '''
        
    # mean loss and mean dice
    mean_loss = float(mean_loss / len(trainloader))
    mean_dice = float(mean_dice / len(trainloader))
    mean_loss_ce = float(mean_loss_ce / len(trainloader))
    writer.add_scalar('info/epoch_train_total_loss', mean_loss, epoch_num)
    writer.add_scalar('info/epoch_train_dice', mean_dice, epoch_num)
    writer.add_scalar('info/epoch_train_total_loss_ce', mean_loss_ce, epoch_num)
    # print('epoch :', epoch_num, 'Train Loss :', mean_loss, 'Train dice :', mean_dice)
    logging.info('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % (epoch_num, mean_loss, mean_loss_ce, mean_dice))

    save_interval = 50  # int(max_epoch/5)
    # if epoch_num > int(max_epoch / 5) and (epoch_num + 1) % save_interval == 0:
    if (epoch_num + 1) >= 350 and (epoch_num + 1) % save_interval == 0:
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num + 1) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

    if epoch_num + 1 == max_epoch:
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num + 1) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
    return writer, mean_dice


def validation(model, writer, dice_weight, ce_weight, args, epoch_num, valloader, snapshot_path):
    model.eval()
    iter_num = epoch_num * len(valloader)
    num_classes = args.num_classes
    class_weights = torch.FloatTensor(ce_weight).cuda()
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(num_classes, dice_weight)
    mean_loss = 0
    mean_dice = 0
    mean_loss_ce = 0
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs= model(image_batch)
            loss_ce_1 = ce_loss(outputs, label_batch[:].long())
            loss_dice_1 = dice_loss(outputs, label_batch, softmax=True)


            loss_ce = loss_ce_1
            loss_dice = loss_dice_1
            dice = 1 - loss_dice_1
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            mean_loss += loss
            mean_dice += dice
            mean_loss_ce += loss_ce

            writer.add_scalar('info/val_total_loss', loss, iter_num)
            writer.add_scalar('info/val_loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/val_dice', dice, iter_num)

            iter_num += 1

            if iter_num % 3 == 0:
                logging.info('epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, val loss_dice: %f, val dice : %f' % (
                    epoch_num, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))



    # mean loss and mean dice
    mean_loss = float(mean_loss / len(valloader))
    mean_dice = float(mean_dice / len(valloader))
    mean_loss_ce = float(mean_loss_ce / len(valloader))
    
    writer.add_scalar('info/epoch_val_total_loss', mean_loss, epoch_num)
    writer.add_scalar('info/epoch_val_dice', mean_dice, epoch_num)
    writer.add_scalar('info/epoch_val_total_loss_ce', mean_loss_ce, epoch_num)
    # print('epoch :', epoch_num, 'validation Loss :', mean_loss, 'validation dice :', mean_dice)
    logging.info('epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f' % (epoch_num, mean_loss, mean_loss_ce, mean_dice))

    return writer, mean_dice


def run_main(args, model, snapshot_path):
    from datasets.dataset import verse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    batch_size = args.batch_size * args.n_gpu
    # random crop 224 * 224 patches from 512 * 512 patches
    db_train = verse_dataset(base_dir=args.train_root_path, list_dir=args.list_dir, split="train", num_classes=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=args.img_size, mode = 'train')]))

    print("The length of train set is: {}".format(len(db_train)))
    # random crop 224 * 224 patches from 512 * 512 patches
    db_val = verse_dataset(base_dir=args.val_root_path, list_dir=args.list_dir, split="val", num_classes=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=args.img_size, mode = 'val')]))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,      # m_worker: 8    ---->>   4
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,      # m_worker: 8    ---->>   4
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
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-5)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 25, eta_min=5e-6)                             
    #  CosineAnnealingWarmRestarts
    # record best model with highest train and val dice
    highest_train_dice = 0
    epo_train = 1
    highest_val_dice = 0
    epo_val = 1
    # these two weight need to be changed
    dice_weight = [0.25] + [0.75] * 25
    ce_weight = [0.25] + [0.75] * 25                # [1.0, 1.0]
    ### train and validation
    for epoch_num in iterator:

        
        writer, train_dice = train(model, writer, optimizer, dice_weight, ce_weight, args, epoch_num, trainloader, snapshot_path)
        writer, val_dice = validation(model, writer, dice_weight, ce_weight, args, epoch_num, valloader, snapshot_path)
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