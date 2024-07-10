#-*- coding:utf-8 -*-

import os
import time
import torch
import datetime
import torch.nn as nn
from torchvision.utils import save_image
# loss
from losses import PerceptualLoss, GANLoss, MultiscaleRecLoss
# util
from utils import Logger, denorm, newnorm, ImagePool, flow_warping, mask_generate, weight_init, rgb2hsv_torch, hsv2rgb_torch
# Enhancement model
from models import Generator, Discriminator, InterMerge
## Optical flow model
from RAFTcore.raft import RAFT
# measurement
from metrics.NIMA.CalcNIMA import NIMA

from metrics.CalcPSNR import calc_psnr
#from metrics.CalcSSIM import calc_ssim
from tqdm import *
# dataloader
from data_loader_outdoor_ab import InputFetcher
# other 
import utils
from pytorch_msssim import ssim
import pyiqa

class Trainer(object):
    def __init__(self, loaders, args):
        # data loader
        self.loaders = loaders
        
        # Model configuration.
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.model_save_path = os.path.join(args.save_root_dir, args.version, args.model_save_path)
        self.sample_path = os.path.join(args.save_root_dir, args.version, args.sample_path)
        self.log_path = os.path.join(args.save_root_dir, args.version, args.log_path)
        self.val_result_path = os.path.join(args.save_root_dir, args.version, args.val_result_path)

        # Build the model and tensorboard.
        self.build_model()
        if self.args.use_tensorboard:
            self.build_tensorboard()

    def train(self):
        """ Train UEGAN ."""
        self.fetcher = InputFetcher(self.loaders.ref)
        self.fetcher_val = InputFetcher(self.loaders.val)

        self.train_steps_per_epoch = len(self.loaders.ref)
        self.model_save_step = int(self.args.model_save_epoch * self.train_steps_per_epoch)

        # set nima, psnr, ssim global parameters
        if self.args.is_test_nima:
            self.best_nima_epoch, self.best_nima = 0, 0.0
        if self.args.is_test_psnr_ssim:
            self.best_psnr_epoch, self.best_psnr = 0, 0.0
            self.best_ssim_epoch, self.best_ssim = 0, 0.0

        # set loss functions 
        self.criterionPercep = PerceptualLoss()
        self.criterionIdt = MultiscaleRecLoss(scale=3, rec_loss_type=self.args.idt_loss_type, multiscale=True)
        self.criterionGAN = GANLoss(self.args.adv_loss_type, tensor=torch.cuda.FloatTensor)
        self.loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
        
        # set quality metric second stage
        brisque_metric = pyiqa.create_metric('brisque').to(self.device)
        nima_model = NIMA()
        nima_model.load_state_dict(torch.load('./metrics/NIMA/pretrain-model.pth'))
        nima_model.to(self.device).eval()
        
        # start from scratch or trained models
        if self.args.pretrained_model:
            start_step = int(self.args.pretrained_model * self.train_steps_per_epoch)
            self.load_pretrained_model(self.args.pretrained_model)
        else:
            start_step = 0
        
        # start training
        print("======================================= start training =======================================")
        self.start_time = time.time()
        total_steps = int(self.args.total_epochs * self.train_steps_per_epoch)
        self.val_start_steps = int(self.args.num_epochs_start_val * self.train_steps_per_epoch)
        self.val_each_steps = int(self.args.val_each_epochs * self.train_steps_per_epoch)
              
        print("=========== start to iteratively train generator and discriminator ===========")
        pbar = tqdm(total=total_steps, desc='Train epoches', initial=start_step)
        for step in range(start_step, total_steps):
            
            
            ########## data iter
            input = next(self.fetcher)
            real_exp, real_raw, self.real_raw_name = input.img_exp, input.img_raw, input.img_name
            
            real_raw_0 = real_raw[0].cuda() 
            real_raw_1 = real_raw[1].cuda() 
            real_raw_2 = real_raw[2].cuda() 
            real_raw_3 = real_raw[3].cuda() 
            real_raw_4 = real_raw[4].cuda() 
            
            real_exp_0 = real_exp[0].cuda() 
            real_exp_1 = real_exp[1].cuda()
            real_exp_2 = real_exp[2].cuda()
            real_exp_3 = real_exp[3].cuda()
            real_exp_4 = real_exp[4].cuda()
            
            [b, c, h, w] = real_raw_2.shape
            
            self.real_raw = torch.cat((real_raw_0, real_raw_1, real_raw_2, real_raw_3, real_raw_4), 0)
            self.real_exp = torch.cat((real_exp_0, real_exp_1, real_exp_2, real_exp_3, real_exp_4), 0)
            
            ########## model train
            self.G.train()
            self.D.train()
            self.T.train()
            self.T_fusion.train()
            self.R.train()
            
            self.fake_exp = self.G(self.real_raw)   
            self.fake_exp_store = self.fake_exp_pool.query(self.fake_exp) 

            ########## update D
            self.d_optimizer.zero_grad()
            real_exp_preds = self.D(self.real_exp)                # [8, 3, 256, 256]
            fake_exp_preds = self.D(self.fake_exp_store.detach()) # [8, 3, 256, 256]
            
            d_loss = self.criterionGAN(real_exp_preds, fake_exp_preds, None, None, for_discriminator=True)
            if self.args.adv_input:
                input_preds = self.D(self.real_raw)
                d_loss += self.criterionGAN(real_exp_preds, input_preds, None, None, for_discriminator=True)
            
            d_loss.backward() 
            self.d_optimizer.step()
            self.d_loss = d_loss.item()

            ########## update G
            self.g_optimizer.zero_grad()
            real_exp_preds = self.D(self.real_exp) # [8, 3, 256, 256] -> [8, 1, 128, 128], [8, 1, 64, 64], [8, 1, 32, 32], [8, 1, 16, 16], [8, 1, 8, 8]
            fake_exp_preds = self.D(self.fake_exp) # [8, 3, 256, 256]
            g_adv_loss = self.args.lambda_adv * self.criterionGAN(real_exp_preds, fake_exp_preds, None, None, for_discriminator=False)
            self.g_adv_loss = g_adv_loss.item()
            g_loss = g_adv_loss
            
            g_percep_loss = self.args.lambda_percep * self.criterionPercep((self.fake_exp+1.)/2., (self.real_raw+1.)/2.)
            self.g_percep_loss = g_percep_loss.item()
            g_loss += g_percep_loss
            
            self.real_exp_idt = self.G(self.real_exp)
            g_idt_loss = self.args.lambda_idt * self.criterionIdt(self.real_exp_idt, self.real_exp)
            self.g_idt_loss = g_idt_loss.item()
            g_loss += g_idt_loss
            
            g_loss.backward(retain_graph=True)
            self.g_optimizer.step()
            self.g_loss = g_loss.item()
            
            ########## update T
            self.t_optimizer.zero_grad()
            self.t_fusion_optimizer.zero_grad()
            self.r_optimizer.zero_grad()
            
            fake_exp_new = self.fake_exp.clone().detach()
            
            [b, c, h, w] = real_raw_2.shape
            
            fake_exp_new_frame0 = denorm(fake_exp_new[0  :  b,:,:,:]) # [-1, 1] -> [0, 1]
            fake_exp_new_frame1 = denorm(fake_exp_new[b  :2*b,:,:,:])
            fake_exp_new_frame2 = denorm(fake_exp_new[2*b:3*b,:,:,:])
            fake_exp_new_frame3 = denorm(fake_exp_new[3*b:4*b,:,:,:])
            fake_exp_new_frame4 = denorm(fake_exp_new[4*b:5*b,:,:,:])
            
            flow_i20 = self.R(fake_exp_new_frame2, fake_exp_new_frame0, self.args.iters, test_mode=False) # [8, 2, 256, 256]
            flow_i21 = self.R(fake_exp_new_frame2, fake_exp_new_frame1, self.args.iters, test_mode=False) 
            flow_i23 = self.R(fake_exp_new_frame2, fake_exp_new_frame3, self.args.iters, test_mode=False)
            flow_i24 = self.R(fake_exp_new_frame2, fake_exp_new_frame4, self.args.iters, test_mode=False)
        
            warp_i0 = flow_warping(fake_exp_new_frame0, flow_i20[-1])   # [8, 3, 256, 256]
            warp_i1 = flow_warping(fake_exp_new_frame1, flow_i21[-1])
            warp_i3 = flow_warping(fake_exp_new_frame3, flow_i23[-1])
            warp_i4 = flow_warping(fake_exp_new_frame4, flow_i24[-1])
            
            warp_i0 = warp_i0.view(b, c, 1, h, w)
            warp_i1 = warp_i1.view(b, c, 1, h, w)
            warp_i3 = warp_i3.view(b, c, 1, h, w)
            warp_i4 = warp_i4.view(b, c, 1, h, w)
            
            frame_input = torch.cat((warp_i0.detach(), warp_i1.detach(),  warp_i1.detach(), warp_i3.detach(), warp_i4.detach()), 2)
            frame_pred = self.T(frame_input)
            frame_i2_rs = fake_exp_new_frame2.view(b, c, 1, h, w)
            frame_target = frame_i2_rs

            frame_input2 = torch.cat((warp_i0.detach(), warp_i1.detach(), frame_pred.detach(), frame_i2_rs,  warp_i3.detach()), 2)
            frame_pred_rf = self.T_fusion(frame_input2) + frame_pred.detach()
            frame_pred_rf = frame_pred_rf.view(b, c, h, w)
            
            flow_i02 = self.R(fake_exp_new_frame0, fake_exp_new_frame2)
            back_warp_i0 = flow_warping(frame_pred_rf, flow_i02[-1])
            
            flow_i12 = self.R(fake_exp_new_frame1, fake_exp_new_frame2)
            back_warp_i1 = flow_warping(frame_pred_rf, flow_i12[-1])
            
            flow_i32 = self.R(fake_exp_new_frame3, fake_exp_new_frame2)
            back_warp_i3 = flow_warping(frame_pred_rf, flow_i32[-1])
            
            flow_i42 = self.R(fake_exp_new_frame4, fake_exp_new_frame2)
            back_warp_i4 = flow_warping(frame_pred_rf, flow_i42[-1])
            
            # raft loss#
            optical_loss = self.loss_fn(warp_i0, frame_i2_rs) + self.loss_fn(warp_i1, frame_i2_rs) + self.loss_fn(warp_i3, frame_i2_rs) + self.loss_fn(warp_i4, frame_i2_rs)
            self.optical_loss = optical_loss.item()
            t_loss = optical_loss
            
            
            
            
            #
            synthesis_loss = self.loss_fn(frame_pred, frame_target) #+ (1 - ssim(frame_pred.view(b, c, h, w), frame_target.view(b, c, h, w), data_range=1.0).cuda() )            
            self.synthesis_loss = synthesis_loss.item()
            t_loss += synthesis_loss
           
            frame_diff = frame_i2_rs - frame_pred
            frame_mask = 1 - torch.exp(-frame_diff*frame_diff/0.01) # 0.00001
            frame_neg_mask = 1 - frame_mask
            
            frame_neg_mask = frame_neg_mask.view(b, c, h, w).detach()
            frame_mask = frame_mask.view(b, c, h, w).detach()
            
            
            #save_image(frame_neg_mask, os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "frame_neg_mask.png"))
            
            #import ipdb
            #ipdb.set_trace()

            #frame_neg_m0, frame_m0 = mask_generate(fake_exp_new_frame0, back_warp_i0, b, c, h, w, 0.01)
            #frame_neg_m1, frame_m1 = mask_generate(fake_exp_new_frame1, back_warp_i1, b, c, h, w, 0.01)
            #frame_neg_m3, frame_m3 = mask_generate(fake_exp_new_frame3, back_warp_i3, b, c, h, w, 0.01)
            #frame_neg_m4, frame_m4 = mask_generate(fake_exp_new_frame4, back_warp_i4, b, c, h, w, 0.01)
            frame_neg_m0, frame_m0 = mask_generate(fake_exp_new_frame0, back_warp_i0, b, c, h, w, 0.01)
            frame_neg_m1, frame_m1 = mask_generate(fake_exp_new_frame1, back_warp_i1, b, c, h, w, 0.01)
            frame_neg_m3, frame_m3 = mask_generate(fake_exp_new_frame3, back_warp_i3, b, c, h, w, 0.01)
            frame_neg_m4, frame_m4 = mask_generate(fake_exp_new_frame4, back_warp_i4, b, c, h, w, 0.01)

            refine_loss = (1/4)*(self.loss_fn(back_warp_i0.view(b, c, h, w)*frame_m0, fake_exp_new_frame0.view(b, c, h, w)*frame_m0) + \
                                 self.loss_fn(back_warp_i1.view(b, c, h, w)*frame_m1, fake_exp_new_frame1.view(b, c, h, w)*frame_m1) + \
                                 self.loss_fn(back_warp_i3.view(b, c, h, w)*frame_m3, fake_exp_new_frame3.view(b, c, h, w)*frame_m3) + \
                                 self.loss_fn(back_warp_i4.view(b, c, h, w)*frame_m4, fake_exp_new_frame4.view(b, c, h, w)*frame_m4)) + \
                                 self.loss_fn(frame_pred_rf*frame_neg_mask, fake_exp_new_frame2*frame_neg_mask) 
                                 
                                 

            self.refine_loss = refine_loss.item()
            t_loss += refine_loss            
             
            t_loss.backward()#retain_graph=True)
            self.t_optimizer.step()
            self.t_fusion_optimizer.step()
            self.r_optimizer.step()
            self.t_loss = t_loss.item()
            
            ########## second 
            ########## update G_second
            self.g_second_optimizer.zero_grad()
            
            self.G_second.train()
            
            frame_pred_rf_second = frame_pred_rf.clone().detach()
            
            frame_i0_second = newnorm(fake_exp_new_frame0.clone().detach()) # [0,1]->[-1,1]
            frame_i1_second = newnorm(fake_exp_new_frame1.clone().detach())
            frame_i2_second = newnorm(frame_pred_rf_second)
            frame_i3_second = newnorm(fake_exp_new_frame3.clone().detach()) 
            frame_i4_second = newnorm(fake_exp_new_frame4.clone().detach())
            
            real_raw_second = torch.cat((frame_i0_second, frame_i1_second, frame_i2_second, frame_i3_second, frame_i4_second), 0)  # [15, 3, 224, 224]
            
            #save_image(denorm(real_raw_second), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "input_video.png"))
            [b, c, h, w] = real_raw_second.shape 
            # predocue the pseudo gt image based on hsv color space
            synthetic_references = self.generate_gt(im=real_raw_second, alpha_lower=1.02, alpha_upper=1.10, number_refs=5) # [15, 5, 3, 224, 224]
            synthetic_references = synthetic_references.view(-1, c, h, w)
            nima_score = torch.zeros(synthetic_references.shape[0] )
            brisque_score = torch.zeros(synthetic_references.shape[0] )
            
                           
            for i in range(synthetic_references.shape[0]):
                preds = nima_model(synthetic_references[i, :, :, :].unsqueeze(0)).data.cpu().numpy()[0]
                for j, e in enumerate(preds, 1):
                    nima_score[i] += j * e
                preds_brisque = brisque_metric(synthetic_references[i, :, :, :].unsqueeze(0))
                brisque_score[i] = preds_brisque
                
            nima_score = nima_score.view(b, -1)    
            brisque_score = brisque_score.view(b, -1)   
            expert_score = nima_score + brisque_score
            min_idx = torch.argmin(expert_score, dim=1) 
            mode_value = torch.mode(min_idx).values.item()
            synthetic_references = synthetic_references.view(b, -1, c, h, w) # [15, 5, 3, 224, 224]                
            gt_img_second = synthetic_references[:, mode_value, :, :, :]
            #save_image(denorm(gt_img_second), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "gt_video.png"))
            
            # forward
            fake_exp_second = self.G_second(real_raw_second)
            g_idt_loss_second = self.args.lambda_idt * self.criterionIdt(fake_exp_second, gt_img_second)   
            g_idt_loss_second.backward()
            self.g_second_optimizer.step()
            self.g_idt_loss_second = g_idt_loss_second.item()
            
            
            ########## update T_second
            self.t_second_optimizer.zero_grad()
            self.t_fusion_second_optimizer.zero_grad()
            self.r_optimizer.zero_grad()
            
            self.T_second.train()
            self.T_fusion_second.train()
            
            
            fake_exp_new_second = gt_img_second.clone().detach()
            
            [b, c, h, w] = frame_i2_second.shape
            
            fake_exp_new_frame0_second = denorm(fake_exp_new_second[0  :  b,:,:,:])
            fake_exp_new_frame1_second = denorm(fake_exp_new_second[b  :2*b,:,:,:])
            fake_exp_new_frame2_second = denorm(fake_exp_new_second[2*b:3*b,:,:,:])
            fake_exp_new_frame3_second = denorm(fake_exp_new_second[3*b:4*b,:,:,:])
            fake_exp_new_frame4_second = denorm(fake_exp_new_second[4*b:5*b,:,:,:])
            
            
            flow_i20_second = self.R(fake_exp_new_frame2_second, fake_exp_new_frame0_second, self.args.iters, test_mode=False) # [8, 2, 256, 256]
            flow_i21_second = self.R(fake_exp_new_frame2_second, fake_exp_new_frame1_second, self.args.iters, test_mode=False) 
            flow_i23_second = self.R(fake_exp_new_frame2_second, fake_exp_new_frame3_second, self.args.iters, test_mode=False)
            flow_i24_second = self.R(fake_exp_new_frame2_second, fake_exp_new_frame4_second, self.args.iters, test_mode=False)
        
            warp_i0_second = flow_warping(fake_exp_new_frame0_second, flow_i20_second[-1])   # [8, 3, 256, 256]
            warp_i1_second = flow_warping(fake_exp_new_frame1_second, flow_i21_second[-1])
            warp_i3_second = flow_warping(fake_exp_new_frame3_second, flow_i23_second[-1])
            warp_i4_second = flow_warping(fake_exp_new_frame4_second, flow_i24_second[-1])
            
            warp_i0_second = warp_i0_second.view(b, c, 1, h, w)
            warp_i1_second = warp_i1_second.view(b, c, 1, h, w)
            warp_i3_second = warp_i3_second.view(b, c, 1, h, w)
            warp_i4_second = warp_i4_second.view(b, c, 1, h, w)
            
            frame_input_second = torch.cat((warp_i0_second.detach(), warp_i1_second.detach(),  warp_i1_second.detach(), warp_i3_second.detach(), warp_i4_second.detach()), 2)
            frame_pred_second = self.T_second(frame_input_second)
            frame_i2_rs_second = fake_exp_new_frame2_second.view(b, c, 1, h, w)
            frame_target_second = frame_i2_rs_second

            frame_input2_second = torch.cat((warp_i0_second.detach(), warp_i1_second.detach(), frame_pred_second.detach(), frame_i2_rs_second,  warp_i3_second.detach()), 2)
            frame_pred_rf_second = self.T_fusion_second(frame_input2_second) + frame_pred_second.detach()
            frame_pred_rf_second = frame_pred_rf_second.view(b, c, h, w)
            
            flow_i02_second = self.R(fake_exp_new_frame0_second, fake_exp_new_frame2_second)
            back_warp_i0_second = flow_warping(frame_pred_rf_second, flow_i02_second[-1])
            
            flow_i12_second = self.R(fake_exp_new_frame1_second, fake_exp_new_frame2_second)
            back_warp_i1_second = flow_warping(frame_pred_rf_second, flow_i12_second[-1])
            
            flow_i32_second = self.R(fake_exp_new_frame3_second, fake_exp_new_frame2_second)
            back_warp_i3_second = flow_warping(frame_pred_rf_second, flow_i32_second[-1])
            
            flow_i42_second = self.R(fake_exp_new_frame4_second, fake_exp_new_frame2_second)
            back_warp_i4_second = flow_warping(frame_pred_rf_second, flow_i42_second[-1])
            
            optical_loss_second = self.loss_fn(warp_i0_second, frame_i2_rs_second) + self.loss_fn(warp_i1_second, frame_i2_rs_second) + self.loss_fn(warp_i3_second, frame_i2_rs_second) + self.loss_fn(warp_i4_second, frame_i2_rs_second)
            self.optical_loss_second = optical_loss_second.item()
            t_loss_second = optical_loss_second
            
            synthesis_loss_second = self.loss_fn(frame_pred_second, frame_target_second) #+ (1 - ssim(frame_pred.view(b, c, h, w), frame_target.view(b, c, h, w), data_range=1.0).cuda() )            
            self.synthesis_loss_second = synthesis_loss_second.item()
            t_loss_second += synthesis_loss_second
           
            frame_diff_second = frame_i2_rs_second - frame_pred_second
            frame_mask_second = 1 - torch.exp(-frame_diff_second*frame_diff_second/0.01) # 0.00001
            frame_neg_mask_second = 1 - frame_mask_second
            
            frame_neg_mask_second = frame_neg_mask_second.view(b, c, h, w).detach()
            frame_mask_second = frame_mask_second.view(b, c, h, w).detach()

            frame_neg_m0_second, frame_m0_second = mask_generate(fake_exp_new_frame0_second, back_warp_i0_second, b, c, h, w, 0.01)
            frame_neg_m1_second, frame_m1_second = mask_generate(fake_exp_new_frame1_second, back_warp_i1_second, b, c, h, w, 0.01)
            frame_neg_m3_second, frame_m3_second = mask_generate(fake_exp_new_frame3_second, back_warp_i3_second, b, c, h, w, 0.01)
            frame_neg_m4_second, frame_m4_second = mask_generate(fake_exp_new_frame4_second, back_warp_i4_second, b, c, h, w, 0.01)

            refine_loss_second = (1/4)*(self.loss_fn(back_warp_i0_second.view(b, c, h, w)*frame_m0_second, fake_exp_new_frame0_second.view(b, c, h, w)*frame_m0_second) + \
                                 self.loss_fn(back_warp_i1_second.view(b, c, h, w)*frame_m1_second, fake_exp_new_frame1_second.view(b, c, h, w)*frame_m1_second) + \
                                 self.loss_fn(back_warp_i3_second.view(b, c, h, w)*frame_m3_second, fake_exp_new_frame3_second.view(b, c, h, w)*frame_m3_second) + \
                                 self.loss_fn(back_warp_i4_second.view(b, c, h, w)*frame_m4_second, fake_exp_new_frame4_second.view(b, c, h, w)*frame_m4_second)) + \
                                 self.loss_fn(frame_pred_rf_second*frame_neg_mask_second, fake_exp_new_frame2_second*frame_neg_mask_second) 
                                 
                                 
            self.refine_loss_second = refine_loss_second.item()
            t_loss_second += refine_loss_second            
             
            t_loss_second.backward()#retain_graph=True)
            self.t_second_optimizer.step()
            self.t_fusion_second_optimizer.step()
            self.r_optimizer.step()
            self.t_loss_second = t_loss_second.item()

            

            ### print info and save models
            self.print_info(step, total_steps, pbar)

            ### logging using tensorboard
            self.logging(step)

            ### validation 
            self.model_validation(step)
            
            ### learning rate update
            if step % self.train_steps_per_epoch == 0:
                current_epoch = step // self.train_steps_per_epoch
                self.lr_scheduler_g.step(epoch=current_epoch)
                self.lr_scheduler_g_second.step(epoch=current_epoch)
                self.lr_scheduler_d.step(epoch=current_epoch)
                
                self.lr_scheduler_t.step(epoch=current_epoch)
                self.lr_scheduler_t_fusion.step(epoch=current_epoch)
                
                self.lr_scheduler_t_second.step(epoch=current_epoch)
                self.lr_scheduler_t_fusion_second.step(epoch=current_epoch)
                
                self.lr_scheduler_r.step(epoch=current_epoch)
                
                for param_group in self.g_optimizer.param_groups:
                    pbar.write("====== Epoch: {:>3d}/{}, Learning rate(lr) of Generator(G): [{}], ".format(((step + 1) // self.train_steps_per_epoch), self.args.total_epochs, param_group['lr']), end='')
                for param_group in self.g_second_optimizer.param_groups:
                    pbar.write("Learning rate (lr) of Generator_second(G2): [{}] ======".format(param_group['lr']))
                for param_group in self.d_optimizer.param_groups:
                    pbar.write("Learning rate (lr) of Discriminator(D): [{}] ======".format(param_group['lr']))
                for param_group in self.t_optimizer.param_groups:
                    pbar.write("Learning rate (lr) of InterMerge(T): [{}] ======".format(param_group['lr']))
                for param_group in self.t_fusion_optimizer.param_groups:
                    pbar.write("Learning rate (lr) of InterMerge(T_fusion): [{}] ======".format(param_group['lr']))
                    
                for param_group in self.t_second_optimizer.param_groups:
                    pbar.write("Learning rate (lr) of InterMerge(T) second: [{}] ======".format(param_group['lr']))
                for param_group in self.t_fusion_second_optimizer.param_groups:
                    pbar.write("Learning rate (lr) of InterMerge(T_fusion) second: [{}] ======".format(param_group['lr']))
                    
                for param_group in self.r_optimizer.param_groups:
                    pbar.write("Learning rate (lr) of Raft(R): [{}] ======".format(param_group['lr']))

            pbar.update(1)
            pbar.set_description(f"Train epoch %.2f" % ((step+1.0)/self.train_steps_per_epoch))
        
        self.val_best_results()

        pbar.write("=========== Complete training ===========")
        pbar.close()

    
    def generate_gt(self, im, alpha_lower, alpha_upper, number_refs):
        B, C, H, W = im.shape
        
        # sampling alpha value
        alpha_values = torch.linspace(alpha_lower, alpha_upper, steps=number_refs).to(im.device)
    
        # HSV color space
        frame_hsv = rgb2hsv_torch(denorm(im))
        frame_v_ori = frame_hsv[:, 2, :, :]#.unsqueeze(1)
        
        
        alpha0 = alpha_values[0]
        beta = torch.tensor(1.0).to(im.device)
        gamma = torch.tensor(1.0).to(im.device)
        dark_frame_v = beta * (alpha0 * frame_v_ori) ** gamma
        frame_hsv[:, 2, :, :] = dark_frame_v
        frame_rgb0 = torch.clamp((hsv2rgb_torch(frame_hsv)-0.5)*2.0, -1.0, 1.0) 
    
        #save_image(denorm(frame_rgb0), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "input_video_generate00.png"))
        
        alpha1 = alpha_values[1]
        beta = torch.tensor(1.0).to(im.device)
        gamma = torch.tensor(1.0).to(im.device)
        dark_frame_v = beta * (alpha1 * frame_v_ori) ** gamma
        frame_hsv[:, 2, :, :] = dark_frame_v
        frame_rgb1 = torch.clamp((hsv2rgb_torch(frame_hsv)- 0.5) * 2.0, -1.0, 1.0) 
    
        #save_image(denorm(frame_rgb1), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "input_video_generate11.png"))
        
        alpha2 = alpha_values[2]
        beta = torch.tensor(1.0).to(im.device)
        gamma = torch.tensor(1.0).to(im.device)
        dark_frame_v = beta * (alpha2 * frame_v_ori) ** gamma
        frame_hsv[:, 2, :, :] = dark_frame_v
        frame_rgb2 = torch.clamp((hsv2rgb_torch(frame_hsv)- 0.5) * 2.0, -1.0, 1.0) 
    
        #save_image(denorm(frame_rgb2), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "input_video_generate22.png"))
        
        alpha3 = alpha_values[3]
        beta = torch.tensor(1.0).to(im.device)
        gamma = torch.tensor(1.0).to(im.device)
        dark_frame_v = beta * (alpha3 * frame_v_ori) ** gamma
        frame_hsv[:, 2, :, :] = dark_frame_v
        frame_rgb3 = torch.clamp((hsv2rgb_torch(frame_hsv)- 0.5) * 2.0, -1.0, 1.0) 
    
        #save_image(denorm(frame_rgb3), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "input_video_generate33.png"))
        
        alpha4 = alpha_values[4]
        beta = torch.tensor(1.0).to(im.device)
        gamma = torch.tensor(1.0).to(im.device)
        dark_frame_v = beta * (alpha4 * frame_v_ori) ** gamma
        frame_hsv[:, 2, :, :] = dark_frame_v
        frame_rgb4 = torch.clamp((hsv2rgb_torch(frame_hsv)- 0.5) * 2.0, -1.0, 1.0) 
    
        #save_image(denorm(frame_rgb4), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "input_video_generate44.png"))
        
        synthetic_reference = torch.cat((frame_rgb0, frame_rgb1, frame_rgb2, frame_rgb3, frame_rgb4), dim=1) 
        
        return synthetic_reference.view(B, number_refs, C, H, W)
    
    def logging(self, step):
        self.loss = {}
        self.images = {}
        
        self.loss['D/Total'] = self.d_loss
        self.loss['G/Total'] = self.g_loss
        self.loss['G_second/Total'] = self.g_idt_loss_second
        self.loss['T/Total'] = self.t_loss
        self.loss['T_second/Total'] = self.t_loss_second
        
        self.loss['G/adv_loss'] = self.g_adv_loss
        self.loss['G/percep_loss'] = self.g_percep_loss
        self.loss['G/idt_loss'] = self.g_idt_loss
        
        self.loss['G_second/idt_loss'] = self.g_idt_loss_second
        
        self.loss['T/optical_loss'] = self.optical_loss
        self.loss['T/refine_loss']  = self.refine_loss
        self.loss['T/synthesis_loss'] = self.synthesis_loss
        
        self.loss['T_second/optical_loss'] = self.optical_loss_second
        self.loss['T_second/refine_loss']  = self.refine_loss_second
        self.loss['T_second/synthesis_loss'] = self.synthesis_loss_second
        
        
        if (step+1) % self.args.log_step == 0:            
            if self.args.use_tensorboard:
                for tag, value in self.loss.items():
                    self.logger.scalar_summary(tag, value, step+1)
                for tag, image in self.images.items():
                    self.logger.images_summary(tag, image, step+1)

    
    def print_info(self, step, total_steps, pbar):
        current_epoch = (step+1) / self.train_steps_per_epoch

        if (step + 1) % self.args.info_step == 0:
            elapsed_num = time.time() - self.start_time
            elapsed = str(datetime.timedelta(seconds=elapsed_num))
            pbar.write("Elapse:{:>.12s}, D_Step:{:>6d}/{}, G_Step:{:>6d}/{}, D_loss:{:>.4f}, G_loss:{:>.4f}, G_percep_loss:{:>.4f}, G_adv_loss:{:>.4f}, G_idt_loss:{:>.4f}".format(elapsed, step + 1, total_steps, (step + 1), total_steps, self.d_loss, self.g_loss, self.g_percep_loss, self.g_adv_loss, self.g_idt_loss)) 
                    
        # save models
        if (step + 1) % self.model_save_step == 0:
            if self.args.parallel:
                if torch.cuda.device_count() > 1:
                    checkpoint = {
                    "G_net": self.G.module.state_dict(),
                    "G_second_net": self.G_second.module.state_dict(),
                    "D_net": self.D.module.state_dict(),
                    "T_net": self.T.module.state_dict(),
                    "T_fusion_net": self.T_fusion.module.state_dict(),
                    
                    "T_second_net": self.T_second.module.state_dict(),
                    "T_fusion_second_net": self.T_fusion_second.module.state_dict(),
                    
                    "R_net": self.R.module.state_dict(),
                
                    "epoch": current_epoch,
                    
                    "g_optimizer": self.g_optimizer.state_dict(),
                    "g_second_optimizer": self.g_second_optimizer.state_dict(),
                    "d_optimizer": self.d_optimizer.state_dict(),
                    "t_optimizer": self.t_optimizer.state_dict(),
                    "t_fusion_optimizer": self.t_fusion_optimizer.state_dict(),
                    
                    "t_second_optimizer": self.t_second_optimizer.state_dict(),
                    "t_fusion_second_optimizer": self.t_fusion_second_optimizer.state_dict(),
                    
                    "r_optimizer": self.r_optimizer.state_dict(),
            
                    "lr_scheduler_g": self.lr_scheduler_g.state_dict(),
                    "lr_scheduler_g_second": self.lr_scheduler_g_second.state_dict(),
                    "lr_scheduler_d": self.lr_scheduler_d.state_dict(),
                    "lr_scheduler_t": self.lr_scheduler_t.state_dict(),
                    "lr_scheduler_t_fusion": self.lr_scheduler_t_fusion.state_dict(),
                    
                    "lr_scheduler_t_second": self.lr_scheduler_t_second.state_dict(),
                    "lr_scheduler_t_fusion_second": self.lr_scheduler_t_fusion_second.state_dict(),
                    
                    "lr_scheduler_r": self.lr_scheduler_r.state_dict()
                     
                    }
            else:
                checkpoint = {
                    "G_net": self.G.state_dict(),
                    "G_second_net": self.G_second.state_dict(),
                    "D_net": self.D.state_dict(),
                    "T_net": self.T.state_dict(),
                    "T_fusion_net": self.T_fusion.state_dict(),
                    
                    "T_second_net": self.T_second.state_dict(),
                    "T_fusion_second_net": self.T_fusion_second.state_dict(),
                    
                    "R_net": self.R.state_dict(),
                    
                    "epoch": current_epoch,
                    
                    "g_optimizer": self.g_optimizer.state_dict(),
                    "g_second_optimizer": self.g_second_optimizer.state_dict(),
                    "d_optimizer": self.d_optimizer.state_dict(),
                    "t_optimizer": self.t_optimizer.state_dict(),
                    "t_fusion_optimizer": self.t_fusion_optimizer.state_dict(),
                    
                    "t_second_optimizer": self.t_second_optimizer.state_dict(),
                    "t_fusion_second_optimizer": self.t_fusion_second_optimizer.state_dict(),
                    
                    "r_optimizer": self.r_optimizer.state_dict(),
                   
                    "lr_scheduler_g": self.lr_scheduler_g.state_dict(),
                    "lr_scheduler_g_second": self.lr_scheduler_g_second.state_dict(),
                    "lr_scheduler_d": self.lr_scheduler_d.state_dict(),
                    "lr_scheduler_t": self.lr_scheduler_t.state_dict(),
                    "lr_scheduler_t_fusion": self.lr_scheduler_t_fusion.state_dict(),
                    
                    "lr_scheduler_t_second": self.lr_scheduler_t_second.state_dict(),
                    "lr_scheduler_t_fusion_second": self.lr_scheduler_t_fusion_second.state_dict(),
                    
                    "lr_scheduler_r": self.lr_scheduler_r.state_dict() 
            
                }
            torch.save(checkpoint, os.path.join(self.model_save_path, '{}_{}_{}.pth'.format(self.args.version, self.args.adv_loss_type, current_epoch)))

            pbar.write("======= Save model checkpoints into {} ======".format(self.model_save_path))              


    def model_validation(self, step):  
        
        if (step + 1) % self.train_steps_per_epoch == 0:
            if (step + 1) % self.val_each_steps == 0:
           
                val = {}
                current_epoch = (step + 1) / self.train_steps_per_epoch
                val_save_path = self.val_result_path + '/' + 'validation_' + str(current_epoch)
                val_compare_save_path = self.val_result_path + '/' + 'validation_compare_' + str(current_epoch)
                val_start = 0
                val_total_steps = len(self.loaders.val)
                
                if not os.path.exists(val_save_path):
                    os.makedirs(val_save_path)
                if not os.path.exists(val_compare_save_path):
                    os.makedirs(val_compare_save_path)
      
                self.G.eval()
                self.G_second.eval()
                self.T.eval()
                self.T_fusion.eval()
                
                self.T_second.eval()
                self.T_fusion_second.eval()
                
                self.R.eval()
               
                pbar = tqdm(total=(val_total_steps - val_start), desc='Validation epoches', position=val_start)
                pbar.write("============================== Start validation ==============================")
                with torch.no_grad():
                    for val_step in range(val_start, val_total_steps):
                        
                        input = next(self.fetcher_val)
                        
                        val_exp_real, val_real_raw, val_name = input.img_exp, input.img_raw, input.img_name
                        
                        val_1 = val_real_raw.cuda()
                        val_2 = val_real_raw.cuda()
                        val_3 = val_real_raw.cuda()
                        val_4 = val_real_raw.cuda()
                        val_5 = val_real_raw.cuda()
                        
                        [b, c, h, w] = val_3.shape
                        
                        val_input = torch.cat((val_1, val_2, val_3, val_4, val_5),0)
                        val_fake_exp = self.G(val_input)  
                        
                        frame_i0 = denorm(val_fake_exp[0  :  b,:,:,:])
                        frame_i1 = denorm(val_fake_exp[b  :2*b,:,:,:])
                        frame_i2 = denorm(val_fake_exp[2*b:3*b,:,:,:])
                        frame_i3 = denorm(val_fake_exp[3*b:4*b,:,:,:])
                        frame_i4 = denorm(val_fake_exp[4*b:5*b,:,:,:])
                        
                        flow_i20 = self.R(frame_i2, frame_i0)         
                        flow_i21 = self.R(frame_i2, frame_i1)
                        flow_i23 = self.R(frame_i2, frame_i3)         
                        flow_i24 = self.R(frame_i2, frame_i4)
                        
                        warp_i0 = flow_warping(frame_i0, flow_i20[-1])
                        warp_i1 = flow_warping(frame_i1, flow_i21[-1])
                        warp_i3 = flow_warping(frame_i3, flow_i23[-1])
                        warp_i4 = flow_warping(frame_i4, flow_i24[-1])
                        
                        warp_i0 = warp_i0.view(b, c, 1, h, w)
                        warp_i1 = warp_i1.view(b, c, 1, h, w)
                        warp_i3 = warp_i3.view(b, c, 1, h, w)
                        warp_i4 = warp_i4.view(b, c, 1, h, w)
                        
                        frame_i2 = frame_i2.view(1, c, 1, h, w)
        
                        frame_input = torch.cat((warp_i0.detach(), warp_i1.detach(), warp_i1.detach(), warp_i3.detach(), warp_i4.detach()), 2)
                        frame_pred = self.T(frame_input)
                                               
                        frame_i2_rs = frame_i2.view(b, c, 1, h, w)        
                        frame_input2 = torch.cat((warp_i0.detach(), warp_i1.detach(), frame_pred.detach(), frame_i2_rs,  warp_i3.detach()), 2)
                        frame_pred_rf = self.T_fusion(frame_input2) + frame_pred.detach()
                        frame_pred_rf = frame_pred_rf.view(b, c, h, w)
                        
                        # second forward
                        real_raw_second = torch.cat((frame_i0.clone().detach(), frame_i1.clone().detach(), frame_pred_rf.clone().detach(), frame_i3.clone().detach(), frame_i4.clone().detach()), 0)  # [15, 3, 224, 224]
                        val_fake_exp_second = self.G_second(newnorm(real_raw_second))

                        frame_i0_second = denorm(val_fake_exp_second[0  :  b,:,:,:])
                        frame_i1_second = denorm(val_fake_exp_second[b  :2*b,:,:,:])
                        frame_i2_second = denorm(val_fake_exp_second[2*b:3*b,:,:,:])
                        frame_i3_second = denorm(val_fake_exp_second[3*b:4*b,:,:,:])
                        frame_i4_second = denorm(val_fake_exp_second[4*b:5*b,:,:,:])
                        
                        
                        flow_i20_second = self.R(frame_i2_second, frame_i0_second)         
                        flow_i21_second = self.R(frame_i2_second, frame_i1_second)
                        flow_i23_second = self.R(frame_i2_second, frame_i3_second)         
                        flow_i24_second = self.R(frame_i2_second, frame_i4_second)
                        
                        warp_i0_second = flow_warping(frame_i0_second, flow_i20_second[-1])
                        warp_i1_second = flow_warping(frame_i1_second, flow_i21_second[-1])
                        warp_i3_second = flow_warping(frame_i3_second, flow_i23_second[-1])
                        warp_i4_second = flow_warping(frame_i4_second, flow_i24_second[-1])
                        
                        warp_i0_second = warp_i0_second.view(b, c, 1, h, w)
                        warp_i1_second = warp_i1_second.view(b, c, 1, h, w)
                        warp_i3_second = warp_i3_second.view(b, c, 1, h, w)
                        warp_i4_second = warp_i4_second.view(b, c, 1, h, w)
                        
                        frame_i2_second = frame_i2_second.view(1, c, 1, h, w)
        
                        frame_input_second = torch.cat((warp_i0_second.detach(), warp_i1_second.detach(), warp_i1_second.detach(), warp_i3_second.detach(), warp_i4_second.detach()), 2)
                        frame_pred_second = self.T_second(frame_input_second)
                                               
                        frame_i2_rs_second = frame_i2_second.view(b, c, 1, h, w)        
                        frame_input2_second = torch.cat((warp_i0_second.detach(), warp_i1_second.detach(), frame_pred_second.detach(), frame_i2_rs_second,  warp_i3_second.detach()), 2)
                        frame_pred_rf_second = self.T_fusion_second(frame_input2_second) + frame_pred_second.detach()
                        frame_pred_rf_second = frame_pred_rf_second.view(b, c, h, w)

        
                        for i in range(0, denorm(frame_pred_rf_second.data).size(0)):
                            save_imgs = frame_pred_rf_second.data[i:i + 1,:,:,:]
                            save_image(save_imgs, os.path.join(val_save_path, '{:s}_{:0>3.2f}_valFakeExp.png'.format(val_name[0], current_epoch)))
                            
                            save_imgs_compare = torch.cat([denorm(val_2.data.cpu())[i:i + 1,:,:,:], frame_pred_rf_second.data.cpu()[i:i + 1,:,:,:]], 3)
                            save_image(save_imgs_compare, os.path.join(val_compare_save_path, '{:s}_{:0>3.2f}_valRealRaw_valFakeExp.png'.format(val_name[0], current_epoch)))
              

                        elapsed = time.time() - self.start_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        if val_step % self.args.info_step == 0:
                            pbar.write("=== Elapse:{}, Save {:>3d}-th val_fake_exp images into {} ===".format(elapsed, val_step, val_save_path))


                        pbar.update(1)

                        if self.args.use_tensorboard:
                            for tag, images in val.items():
                                self.logger.images_summary(tag, images, val_step + 1)
                    
                    pbar.close()
                    if self.args.is_test_nima:
                        self.nima_result_save_path = './results/nima_val_results/'
                        curr_nima = calc_nima(val_save_path, self.nima_result_save_path,  current_epoch)
                        if self.best_nima < curr_nima:
                            self.best_nima = curr_nima
                            self.best_nima_epoch = current_epoch
                        print("====== Avg. NIMA: {:>.4f} ======".format(curr_nima))
                    
                    if self.args.is_test_psnr_ssim:
                        self.psnr_save_path = './results/psnr_val_results/' 
                        curr_psnr = calc_psnr(val_save_path, self.args.img_dir_SDSD, self.psnr_save_path, current_epoch)
                        if self.best_psnr < curr_psnr:
                            self.best_psnr = curr_psnr
                            self.best_psnr_epoch = current_epoch
                        print("====== Avg. PSNR: {:>.4f} dB ======".format(curr_psnr))
      
                        self.ssim_save_path = './results/ssim_val_results/' 
                        curr_ssim = calc_ssim(val_save_path, self.args.img_dir_SDSD, self.ssim_save_path, current_epoch)
                        if self.best_ssim < curr_ssim:
                            self.best_ssim = curr_ssim
                            self.best_ssim_epoch = current_epoch
                        print("====== Avg. SSIM: {:>.4f}  ======".format(curr_ssim))
                    torch.cuda.empty_cache()
                    time.sleep(2)    
                            

    def val_best_results(self):
        if self.args.is_test_psnr_ssim:
            if not os.path.exists(self.psnr_save_path):
                os.makedirs(self.psnr_save_path)
            psnr_result = self.psnr_save_path + 'PSNR_total_results_epoch_avgpsnr.csv'
            psnrfile = open(psnr_result, 'a+')
            psnrfile.write('Best epoch: ' + str(self.best_psnr_epoch) + ',' + str(round(self.best_psnr, 6)) + '\n')
            psnrfile.close()

            if not os.path.exists(self.ssim_save_path):
                os.makedirs(self.ssim_save_path)
            ssim_result = self.ssim_save_path + 'SSIM_total_results_epoch_avgssim.csv'
            ssimfile = open(ssim_result, 'a+')
            ssimfile.write('Best epoch: ' + str(self.best_ssim_epoch) + ',' + str(round(self.best_ssim, 6)) + '\n')
            ssimfile.close()

        if self.args.is_test_nima:
            nima_total_result = self.nima_result_save_path + 'NIMA_total_results_epoch_mean_std.csv'
            totalfile = open(nima_total_result, 'a+')
            totalfile.write('Best epoch:' + str(self.best_nima_epoch) + ',' + str(round(self.best_nima, 6)) + '\n')
            totalfile.close()

    
    """define some functions"""
    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.args.g_conv_dim, self.args.g_norm_fun, self.args.g_act_fun, self.args.g_use_sn).to(self.device)
        self.G_second = Generator(self.args.g_conv_dim, self.args.g_norm_fun, self.args.g_act_fun, self.args.g_use_sn).to(self.device)
        self.D = Discriminator(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
        
        self.T = InterMerge(self.args.t_conv_dim).to(self.device)
        self.T_second = InterMerge(self.args.t_conv_dim).to(self.device)
        
        self.T_fusion = InterMerge(self.args.t_conv_dim).to(self.device)
        self.T_fusion_second = InterMerge(self.args.t_conv_dim).to(self.device)
        
        self.R = RAFT(self.args).to(self.device)
        
        if self.args.parallel:
            self.G.to(self.args.gpu_ids[0])
            self.G_second.to(self.args.gpu_ids[0])
            
            self.D.to(self.args.gpu_ids[0])
            
            self.T.to(self.args.gpu_ids[0])
            self.T_second.to(self.args.gpu_ids[0])
            
            self.T_fusion.to(self.args.gpu_ids[0])
            self.T_fusion_second.to(self.args.gpu_ids[0])
            
            self.R.to(self.args.gpu_ids[0])
          
            self.G = nn.DataParallel(self.G, self.args.gpu_ids)
            self.G_second = nn.DataParallel(self.G_second, self.args.gpu_ids)
            self.D = nn.DataParallel(self.D, self.args.gpu_ids)
            
            self.T = nn.DataParallel(self.T, self.args.gpu_ids)
            self.T_second = nn.DataParallel(self.T_second, self.args.gpu_ids)

            self.T_fusion = nn.DataParallel(self.T_fusion, self.args.gpu_ids)
            self.T_fusion_second = nn.DataParallel(self.T_fusion_second, self.args.gpu_ids)
            
            self.R = nn.DataParallel(self.R, self.args.gpu_ids)
           
        print("=== Models have been created ===")
        
        # print network
        if self.args.is_print_network:
            self.print_network(self.G, 'Generator')
            self.print_network(self.G_second, 'Generator_second')
            self.print_network(self.D, 'Discriminator')
            
            self.print_network(self.T, 'InterMerge')
            self.print_network(self.T_second, 'InterMerge_second')
            
            self.print_network(self.T_fusion, 'InterMerge')
            self.print_network(self.T_fusion_second, 'InterMerge_second')
            
            self.print_network(self.R, 'Raft')
           
    
        # init network
        if self.args.init_type:
            self.init_weights(self.G, init_type=self.args.init_type, gain=0.02)
            self.init_weights(self.G_second, init_type=self.args.init_type, gain=0.02)
            self.init_weights(self.D, init_type=self.args.init_type, gain=0.02)
                        
            self.T.apply(weight_init)
            self.T_fusion.apply(weight_init)
            self.T_second.apply(weight_init)
            self.T_fusion_second.apply(weight_init)
            
        #if self.args.finetune_raft:
        #    self.R.load_state_dict(torch.load(self.args.raft_model_path))
        #    print("=== Pretrained Raft Model have been Loaded ===")
        if self.args.finetune_raft:
            self.R.load_state_dict({k.replace('module.',''):v for k,v in torch.load(self.args.raft_model_path).items()})
            print("=== Pretrained Raft Model have been Loaded ===")
           
        # optimizer
        if self.args.optimizer_type == 'adam':
            # Adam optimizer
            self.g_optimizer = torch.optim.Adam(params=self.G.parameters(), lr=self.args.g_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            self.g_second_optimizer = torch.optim.Adam(params=self.G_second.parameters(), lr=self.args.g_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            self.d_optimizer = torch.optim.Adam(params=self.D.parameters(), lr=self.args.d_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            
            self.t_fusion_optimizer = torch.optim.Adam(params=self.T_fusion.parameters(), lr=self.args.t_fusion_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            self.t_optimizer = torch.optim.Adam(params=self.T.parameters(), lr=self.args.t_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            
            self.t_fusion_second_optimizer = torch.optim.Adam(params=self.T_fusion_second.parameters(), lr=self.args.t_fusion_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            self.t_second_optimizer = torch.optim.Adam(params=self.T_second.parameters(), lr=self.args.t_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            
            self.r_optimizer = torch.optim.Adam(params=self.R.parameters(), lr=self.args.r_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            
        elif self.args.optimizer_type == 'rmsprop':
            # RMSprop optimizer
            self.g_optimizer = torch.optim.RMSprop(params=self.G.parameters(), lr=self.args.g_lr, alpha=self.args.alpha)
            self.g_second_optimizer = torch.optim.RMSprop(params=self.G_second.parameters(), lr=self.args.g_lr, alpha=self.args.alpha)
            self.d_optimizer = torch.optim.RMSprop(params=self.D.parameters(), lr=self.args.d_lr, alpha=self.args.alpha)
            
            self.t_fusion_optimizer = torch.optim.RMSprop(params=self.T_fusion.parameters(), lr=self.args.t_fusion_lr, alpha=self.args.alpha)
            self.t_optimizer = torch.optim.RMSprop(params=self.T.parameters(), lr=self.args.t_lr, alpha=self.args.alpha)
            
            self.t_fusion_second_optimizer = torch.optim.RMSprop(params=self.T_fusion_second.parameters(), lr=self.args.t_fusion_lr, alpha=self.args.alpha)
            self.t__second_optimizer = torch.optim.RMSprop(params=self.T_second.parameters(), lr=self.args.t_lr, alpha=self.args.alpha)
            
            self.r_optimizer = torch.optim.RMSprop(params=self.R.parameters(), lr=self.args.r_lr, alpha=self.args.alpha)
            
        else:
            raise NotImplementedError("=== Optimizer [{}] is not found ===".format(self.args.optimizer_type))

        # learning rate decay
        if self.args.lr_decay:
            def lambda_rule(epoch):
                return 1.0 - max(0, epoch + 1 - self.args.lr_num_epochs_decay) / self.args.lr_decay_ratio
            self.lr_scheduler_g = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda_rule)
            self.lr_scheduler_g_second = torch.optim.lr_scheduler.LambdaLR(self.g_second_optimizer, lr_lambda=lambda_rule)
            self.lr_scheduler_d = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=lambda_rule)
            
            self.lr_scheduler_t = torch.optim.lr_scheduler.LambdaLR(self.t_optimizer, lr_lambda=lambda_rule)
            self.lr_scheduler_t_fusion = torch.optim.lr_scheduler.LambdaLR(self.t_fusion_optimizer, lr_lambda=lambda_rule)
            
            self.lr_scheduler_t_second = torch.optim.lr_scheduler.LambdaLR(self.t_second_optimizer, lr_lambda=lambda_rule)
            self.lr_scheduler_t_fusion_second = torch.optim.lr_scheduler.LambdaLR(self.t_fusion_second_optimizer, lr_lambda=lambda_rule)
            
            self.lr_scheduler_r = torch.optim.lr_scheduler.LambdaLR(self.r_optimizer, lr_lambda=lambda_rule)
           
            print("=== Set learning rate decay policy for Generator(G), Discriminator(D), Merge(T) and Raft(R) ===")
        
        self.fake_exp_pool = ImagePool(self.args.pool_size)
        

    def init_weights(self, net, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('Initialization method [{}] is not implemented'.format(init_type))
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data,   0.0)
        print("=== Initialize network with [{}] ===".format(init_type))
        net.apply(init_func)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print("=== The number of parameters of the above model [{}] is [{}] or [{:>.4f}M] ===".format(name, num_params, num_params / 1e6))


    def load_pretrained_model(self, resume_epochs):
        checkpoint_path = os.path.join(self.model_save_path, '{}_{}_{}.pth'.format(self.args.version, self.args.adv_loss_type, resume_epochs))
        if torch.cuda.is_available():
            # save on GPU, load on GPU
            checkpoint = torch.load(checkpoint_path)
            self.G.load_state_dict(checkpoint['G_net'])
            self.G_second.load_state_dict(checkpoint['G_second_net'])
            self.D.load_state_dict(checkpoint['D_net'])
            
            self.T_second.load_state_dict(checkpoint['T_second_net'])
            self.T_fusion_second.load_state_dict(checkpoint['T_fusion_second_net'])
            
            self.T.load_state_dict(checkpoint['T_net'])
            self.T_fusion.load_state_dict(checkpoint['T_fusion_net'])
            
            self.R.load_state_dict(checkpoint['R_net'])
           
            
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.g_second_optimizer.load_state_dict(checkpoint['g_second_optimizer'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            
            self.t_optimizer.load_state_dict(checkpoint['t_optimizer'])
            self.t_fusion_optimizer.load_state_dict(checkpoint['t_fusion_optimizer'])
            
            self.t_second_optimizer.load_state_dict(checkpoint['t_second_optimizer'])
            self.t_fusion_second_optimizer.load_state_dict(checkpoint['t_fusion_second_optimizer'])
            
            self.r_optimizer.load_state_dict(checkpoint['r_optimizer'])
          
            
            self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g'])
            self.lr_scheduler_g_second.load_state_dict(checkpoint['lr_scheduler_g_second'])
            self.lr_scheduler_d.load_state_dict(checkpoint['lr_scheduler_d'])
            
            self.lr_scheduler_t_second.load_state_dict(checkpoint['lr_scheduler_t_second'])
            self.lr_scheduler_t_fusion_second.load_state_dict(checkpoint['lr_scheduler_t_fusion_second'])
            
            self.lr_scheduler_t.load_state_dict(checkpoint['lr_scheduler_t'])
            self.lr_scheduler_t_fusion.load_state_dict(checkpoint['lr_scheduler_t_fusion'])
            
            
            self.lr_scheduler_r.load_state_dict(checkpoint['lr_scheduler_r'])
           
            
        else:
            # save on GPU, load on CPU
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            self.G.load_state_dict(checkpoint['G_net'])
            self.G_second.load_state_dict(checkpoint['G_second_net'])
            self.D.load_state_dict(checkpoint['D_net'])
            self.T.load_state_dict(checkpoint['T_net'])
            self.T_fusion.load_state_dict(checkpoint['T_fusion_net'])
            
            self.T_second.load_state_dict(checkpoint['T_second_net'])
            self.T_fusion_second.load_state_dict(checkpoint['T_fusion_second_net'])
            
            self.R.load_state_dict(checkpoint['R_net'])
            
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.g_second_optimizer.load_state_dict(checkpoint['g_second_optimizer'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            
            self.t_optimizer.load_state_dict(checkpoint['t_optimizer'])
            self.t_fusion_optimizer.load_state_dict(checkpoint['t_fusion_optimizer'])
            
            self.t_second_optimizer.load_state_dict(checkpoint['t_second_optimizer'])
            self.t_fusion_second_optimizer.load_state_dict(checkpoint['t_fusion_second_optimizer'])
            
            self.r_optimizer.load_state_dict(checkpoint['r_optimizer'])
            
            self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g'])
            self.lr_scheduler_g_second.load_state_dict(checkpoint['lr_scheduler_g_second'])
            self.lr_scheduler_d.load_state_dict(checkpoint['lr_scheduler_d'])
            
            self.lr_scheduler_t.load_state_dict(checkpoint['lr_scheduler_t'])
            self.lr_scheduler_t_fusion.load_state_dict(checkpoint['lr_scheduler_t_fusion'])
            
            self.lr_scheduler_t_second.load_state_dict(checkpoint['lr_scheduler_t_second'])
            self.lr_scheduler_t_fusion_second.load_state_dict(checkpoint['lr_scheduler_t_fusion_second'])
            
            self.lr_scheduler_r.load_state_dict(checkpoint['lr_scheduler_r'])
           
        print("=========== loaded trained models (epochs: {})! ===========".format(resume_epochs))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_path)
    
    def identity_loss(self, idt_loss_type):
        if idt_loss_type == 'l1':
            criterion = nn.L1Loss()
            return criterion
        elif idt_loss_type == 'smoothl1':
            criterion = nn.SmoothL1Loss()
            return criterion
        elif idt_loss_type == 'l2':
            criterion = nn.MSELoss()
            return criterion
        else:
            raise NotImplementedError("=== Identity loss type [{}] is not implemented. ===".format(self.args.idt_loss_type))
  