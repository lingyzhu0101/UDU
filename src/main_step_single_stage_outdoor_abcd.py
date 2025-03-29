#-*-coding:utf-8-*-

import os
import argparse
from trainer_single_stage_outdoor_abcd import Trainer
from tester_single_stage_outdoor_abcd import Tester
#from tester_single_stage_outdoor_abcd_cross import Tester
from utils import create_folder, setup_seed
from config_ab import get_config
import torch
from munch import Munch
from data_loader_outdoor_ab import get_train_loader, get_val_loader


def main(args):
    # for fast training.
    torch.backends.cudnn.benchmark = True

    setup_seed(args.seed)
    
    # create directories if not exist.
    create_folder(args.save_root_dir, args.version, args.model_save_path)
    create_folder(args.save_root_dir, args.version, args.sample_path)
    create_folder(args.save_root_dir, args.version, args.log_path)
    create_folder(args.save_root_dir, args.version, args.val_result_path)
    create_folder(args.save_root_dir, args.version, args.test_result_path)

    if args.mode == 'train':
        loaders = Munch(ref=get_train_loader(root=args.img_dir_SDSD,
                                            sample_frames=args.sample_frames, 
                                            img_size=args.image_size,
                                            resize_size=args.resize_size,
                                            batch_size=args.train_batch_size,
                                            shuffle=args.shuffle,
                                            num_workers=args.num_workers,
                                            drop_last=True),
                        val=get_val_loader(root=args.img_dir_SDSD,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        trainer = Trainer(loaders, args)
        #trainer.model_validation(step=1)
        trainer.train()
        
    elif args.mode == 'test':
        tester = Tester(args)
        tester.test()
    else:
        raise NotImplementedError('Mode [{}] is not found'.format(args.mode))


if __name__ == '__main__':

    args = get_config()
    main(args)