

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.MedNextV1 import MedNeXt
from trainer import run_main

parser = argparse.ArgumentParser()
parser.add_argument('--train_root_path', type=str,
                    default='xxxxxx/verse19', help='root dir for train data')
parser.add_argument('--val_root_path', type=str,
                    default='xxxxxx/verse19', help='root dir for val data')
parser.add_argument('--dataset', type=str,
                    default='verse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='xxxxx/lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=26, help='output channel of network')        # 25 + 1
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=1000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')             # 24
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,        # 0.01
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=[128, 160, 96], help='input patch size of network input')                 # network input image size : 128
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    dataset_name = args.dataset
    dataset_config = {
        'verse': {
            'train_root_path': 'xxxxx/verse19',
            'val_root_path': 'xxxxxx/verse19',
            'list_dir': 'xxxxx/lists/lists_Synapse',
            'num_classes': 26,
        },
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.train_root_path = dataset_config[dataset_name]['train_root_path']
    args.val_root_path = dataset_config[dataset_name]['val_root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.batch_size = args.batch_size
    # number of patches
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size[0] / args.vit_patches_size), int(args.img_size[1] / args.vit_patches_size), int(args.img_size[2] / args.vit_patches_size))
    ###
    config_vit.n_patches = int(args.img_size[0] / args.vit_patches_size) * int(args.img_size[1] / args.vit_patches_size) * int(args.img_size[2] / args.vit_patches_size)
    config_vit.n_patches = int(args.img_size[0] / args.vit_patches_size) * int(args.img_size[1] / args.vit_patches_size) * int(args.img_size[2] / args.vit_patches_size)
    config_vit.h = int(args.img_size[0] / args.vit_patches_size)
    config_vit.w = int(args.img_size[1] / args.vit_patches_size)
    config_vit.l = int(args.img_size[2] / args.vit_patches_size)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'         # '0,1,2,3'
    torch.cuda.empty_cache()
    # net = Net(in_channel=3, out_channel=args.num_classes, training=False, config=config_vit).cuda()
    # large
    net = MedNeXt(
        in_channels = 1, 
        n_channels = 32,
        n_classes = args.num_classes,
        exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
        # exp_r = 2,
        kernel_size=3,                     # Can test kernel_size
        deep_supervision=False,             # Can be used to test deep supervision
        do_res=True,                      # Can be used to individually test residual connection
        do_res_up_down = True,
        # block_counts = [2,2,2,2,2,2,2,2,2],
        block_counts = [3,4,4,4,4,4,4,4,3],
        checkpoint_style = 'outside_block'
    ).cuda()
    # load pretrained weight
    # net.load_from(weights=np.load(config_vit.pretrained_path))


    # train the network
    transunet = {'verse': run_main,}
    transunet[dataset_name](args, net, snapshot_path)

