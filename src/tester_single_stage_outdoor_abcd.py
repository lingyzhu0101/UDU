#-*- coding:utf-8 -*-

import os, cv2, glob, utils
import numpy as np
import time
import torch
import datetime
import torch.nn as nn
from torchvision.utils import save_image
# loss
from losses import PerceptualLoss, TVLoss
# utils
from utils import Logger, denorm, newnorm, ImagePool, GaussianNoise, rgb2hsv_torch, hsv2rgb_torch
# model
from models import Generator, Discriminator, InterMerge
## Optical flow model
from RAFTcore.raft import RAFT
# measurement
from metrics.CalcPSNR import calc_psnr
#from metrics.CalcSSIM import calc_ssim
from tqdm import *
from data_loader_outdoor_ab import InputFetcher
#utils
from utils import Logger, denorm, ImagePool, flow_warping, mask_generate

# quality measure
import pyiqa
from metrics.NIMA.CalcNIMA import NIMA

class Tester(object):
    def __init__(self, args):

        # Model configuration.
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.model_save_path = os.path.join(args.save_root_dir, args.version, args.model_save_path)
        self.sample_path = os.path.join(args.save_root_dir, args.version, args.sample_path)
        self.log_path = os.path.join(args.save_root_dir, args.version, args.log_path)
        #self.test_result_path = os.path.join(args.save_root_dir, args.version, args.test_result_path)
        self.test_result_path = os.path.join(args.save_root_dir, args.version, args.test_result_path + "_epoch_" + str(self.args.pretrained_model))

        # Build the model and tensorboard.
        self.build_model()
        if self.args.use_tensorboard:
            self.build_tensorboard()

    def test(self):    
        """ Test UEGAN ."""
        start_time = time.time()
        test_start = 0
        
        
        # set quality metric second stage
        brisque_metric = pyiqa.create_metric('brisque').to(self.device)
        nima_model = NIMA()
        nima_model.load_state_dict(torch.load('./metrics/NIMA/pretrain-model.pth'))
        nima_model.to(self.device).eval()
        
        
        self.load_pretrained_model(self.args.pretrained_model)
        self.G.eval()
        self.G_second.eval()
        self.R.eval()
        self.T.eval()
        self.T_second.eval()
        self.T_fusion.eval()
        self.T_fusion_second.eval()

        print("======================================= start testing =========================================")
        test_ids = [line.rstrip('\n') for line in open(self.args.img_dir_SDSD + './SDSD_CUHK/test_list_outdoor.txt')]   # train_list_outdoor_five_video   test_list_outdoor
        
        for test_id in test_ids:
            test_result_path_frames_a = self.test_result_path + "_a"+ "/frames/%s"%test_id
            test_result_path_frames_ab = self.test_result_path + "_ab"+ "/frames/%s"%test_id
            test_result_path_frames_abc = self.test_result_path + "_abc"+ "/frames/%s"%test_id
            test_result_path_frames_abcd = self.test_result_path + "_abcd"+ "/frames/%s"%test_id
            
            
            test_result_path_frames_wo_human = self.test_result_path + "_wo_human"+ "/frames/%s"%test_id
            
            if not os.path.exists(test_result_path_frames_a):
                os.makedirs(test_result_path_frames_a)
                
            if not os.path.exists(test_result_path_frames_ab):
                os.makedirs(test_result_path_frames_ab)
                
            if not os.path.exists(test_result_path_frames_abc):
                os.makedirs(test_result_path_frames_abc)
                
            if not os.path.exists(test_result_path_frames_abcd):
                os.makedirs(test_result_path_frames_abcd)
                
            if not os.path.exists(test_result_path_frames_wo_human):
                os.makedirs(test_result_path_frames_wo_human)
            
            frame_list = sorted(glob.glob(os.path.join(self.args.img_dir_SDSD, "SDSD_CUHK/videoSDSD_low_resize", test_id, '*.*')))
            
            
            for t in range(2, len(frame_list)-2):
            
                test_real_raw0, _ = utils.read_img(os.path.join(self.args.img_dir_SDSD, "SDSD_CUHK/videoSDSD_low_resize", test_id, "%05d.png" % (t-2)))
                test_real_raw1, _ = utils.read_img(os.path.join(self.args.img_dir_SDSD, "SDSD_CUHK/videoSDSD_low_resize", test_id, "%05d.png" % (t-1)))
                test_real_raw2, _ = utils.read_img(os.path.join(self.args.img_dir_SDSD, "SDSD_CUHK/videoSDSD_low_resize", test_id, "%05d.png" % (t)))
                test_real_raw3, _ = utils.read_img(os.path.join(self.args.img_dir_SDSD, "SDSD_CUHK/videoSDSD_low_resize", test_id, "%05d.png" % (t+1)))
                test_real_raw4, _ = utils.read_img(os.path.join(self.args.img_dir_SDSD, "SDSD_CUHK/videoSDSD_low_resize", test_id, "%05d.png" % (t+2)))
                
                with torch.no_grad():
                    test_real_raw0 = utils.img2tensor(test_real_raw0).cuda()
                    test_real_raw1 = utils.img2tensor(test_real_raw1).cuda()
                    test_real_raw2 = utils.img2tensor(test_real_raw2).cuda()
                    test_real_raw3 = utils.img2tensor(test_real_raw3).cuda()
                    test_real_raw4 = utils.img2tensor(test_real_raw4).cuda()
                    
                    [b, c, h, w] = test_real_raw0.shape
                    
                    frame_input2 = torch.cat((test_real_raw0, test_real_raw1, test_real_raw2, test_real_raw3, test_real_raw4), 0)
                    
                    #start_time_G = time.time()
                    
                    fake_exp_new = self.G(frame_input2)
                    
                    
                    #elapsed_G = time.time() - start_time_G
                    #elapsed = str(datetime.timedelta(seconds=elapsed_G))
                    #print("=== Elapse:{} ===".format(elapsed))  
                
                    
                    frame_i0 = denorm(fake_exp_new[0  :  b,:,:,:])
                    frame_i1 = denorm(fake_exp_new[b  :2*b,:,:,:])
                    frame_i2 = denorm(fake_exp_new[2*b:3*b,:,:,:])
                    frame_i3 = denorm(fake_exp_new[3*b:4*b,:,:,:])
                    frame_i4 = denorm(fake_exp_new[4*b:5*b,:,:,:])
                       
                    
                    flow_i20 = self.R(frame_i2, frame_i0)         
                    flow_i21 = self.R(frame_i2, frame_i1)
                    flow_i23 = self.R(frame_i2, frame_i3)         
                    flow_i24 = self.R(frame_i2, frame_i4)
                    
                    
                    warp_i0 = flow_warping(frame_i0, flow_i20[-1])
                    warp_i1 = flow_warping(frame_i1, flow_i21[-2])
                    warp_i3 = flow_warping(frame_i3, flow_i23[-1])
                    warp_i4 = flow_warping(frame_i4, flow_i24[-2])
                    
                    warp_i0 = warp_i0.view(b, c, 1, h, w)
                    warp_i1 = warp_i1.view(b, c, 1, h, w)
                    warp_i3 = warp_i3.view(b, c, 1, h, w)
                    warp_i4 = warp_i4.view(b, c, 1, h, w)
                    
                    frame_i2 = frame_i2.view(1, c, 1, h, w)
    
                    frame_input = torch.cat((warp_i0.detach(), warp_i1.detach(), warp_i1.detach(), warp_i3.detach(), warp_i4.detach()), 2)
                    
                    
                    #start_time_T = time.time()
                    
                    frame_pred = self.T(frame_input)
                    
                    #elapsed_T = time.time() - start_time_T
                    #elapsed = str(datetime.timedelta(seconds=elapsed_T))
                    #print("=== Elapse  T : {} ===".format(elapsed))  
                    
                    
                                           
                    frame_i2_rs = frame_i2.view(b, c, 1, h, w)        
                    frame_input2 = torch.cat((warp_i0.detach(), warp_i1.detach(), frame_pred.detach(), frame_i2_rs,  warp_i3.detach()), 2)
                    frame_pred_rf = self.T_fusion(frame_input2) + frame_pred.detach()
                    frame_pred_rf = frame_pred_rf.view(b, c, h, w)
                    
                    

                    
                    
                    #synthetic_references = self.generate_gt(im=newnorm(frame_pred_rf), alpha_lower=1.00, alpha_upper=2.0, number_refs=5) # [15, 5, 3, 224, 224]
                    #synthetic_references= synthetic_references.unsqueeze(0)
                    
                    #synthetic_references = synthetic_references.view(-1, c, h, w)
                    #nima_score = torch.zeros(synthetic_references.shape[0] )
                    #brisque_score = torch.zeros(synthetic_references.shape[0] )
                    
                                   
                    #for i in range(synthetic_references.shape[0]):
                    #    preds = nima_model(synthetic_references[i, :, :256, :256].unsqueeze(0)).data.cpu().numpy()[0]
                    #    for j, e in enumerate(preds, 1):
                    #        nima_score[i] += j * e
                    #    preds_brisque = brisque_metric(synthetic_references[i, :, :, :].unsqueeze(0))
                    #    brisque_score[i] = preds_brisque
                        
                        
                    #import ipdb
                    #ipdb.set_trace()
                    
                    
                    
                    
                    
                    
                    real_raw_second = torch.cat((frame_i0.clone().detach(), frame_i1.clone().detach(), frame_pred_rf.clone().detach(), frame_i3.clone().detach(), frame_i4.clone().detach()), 0)  # [15, 3, 224, 224]
                    val_fake_exp_second = self.G_second(newnorm(real_raw_second))
  
                    frame_i0_second = denorm(val_fake_exp_second[0  :  b,:,:,:])
                    frame_i1_second = denorm(val_fake_exp_second[b  :2*b,:,:,:])
                    frame_i2_second = denorm(val_fake_exp_second[2*b:3*b,:,:,:])
                    #frame_i2_second = frame_pred_rf 
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
                    
                    #for i in range(0, frame_pred_rf_second.data.size(0)):
                    #    save_imgs = frame_pred_rf_second.data[i:i + 1,:,:,:]
                    #    save_image(save_imgs, os.path.join(test_result_path_frames_wo_human, '%05d.png'%t))     

                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("=== Elapse:{}, Save test_fake_exp===".format(elapsed))     
                    
                    
                    #_, flow_up_12 = self.Raft_model(frame_first, frame_second, iters=12, test_mode=True)
                    #warp_frame2 = flow_warping(frame_second, flow_up_12)                
                    #noc_mask2 = torch.exp(- 50 * torch.sum(frame_first - warp_frame2, dim=1).pow(2) ).unsqueeze(1)
                    
                    #frame_first_pred = utils.tensor2img(frame_first) 
                    #frame_second_pred = utils.tensor2img(frame_second) 
                    #warp_frame2 = utils.tensor2img(warp_frame2) 
                    #noc_mask2 = utils.tensor2img(noc_mask2) 
                    
                    #flow_up_12 = flow_up_12[0].permute(1,2,0).cpu().numpy()
                    #flow_up_12 = flow_to_image(flow_up_12)
                    #cv2.imwrite(os.path.join(self.test_result_path_flo, '%05d.png'%t), flow_up_12)
                    
                    #utils.save_img(frame_first_pred, os.path.join(self.test_result_path_frames, '%05d.png'%t))
                    #utils.save_img(frame_second_pred, os.path.join(self.test_result_path_frames, '%05d.png'%(t+1)))
                    
                    #utils.save_img(warp_frame2, os.path.join(self.test_result_path_warp, '%05d.png'%t))
                    #utils.save_img(noc_mask2, os.path.join(self.test_result_path_mask, '%05d.png'%t))  
            
            
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
    
        save_image(denorm(frame_rgb0), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "input_video_generate00.png"))
        
        alpha1 = alpha_values[1]
        beta = torch.tensor(1.0).to(im.device)
        gamma = torch.tensor(1.0).to(im.device)
        dark_frame_v = beta * (alpha1 * frame_v_ori) ** gamma
        frame_hsv[:, 2, :, :] = dark_frame_v
        frame_rgb1 = torch.clamp((hsv2rgb_torch(frame_hsv)- 0.5) * 2.0, -1.0, 1.0) 
    
        save_image(denorm(frame_rgb1), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "input_video_generate11.png"))
        
        alpha2 = alpha_values[2]
        beta = torch.tensor(1.0).to(im.device)
        gamma = torch.tensor(1.0).to(im.device)
        dark_frame_v = beta * (alpha2 * frame_v_ori) ** gamma
        frame_hsv[:, 2, :, :] = dark_frame_v
        frame_rgb2 = torch.clamp((hsv2rgb_torch(frame_hsv)- 0.5) * 2.0, -1.0, 1.0) 
    
        save_image(denorm(frame_rgb2), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "input_video_generate22.png"))
        
        alpha3 = alpha_values[3]
        beta = torch.tensor(1.0).to(im.device)
        gamma = torch.tensor(1.0).to(im.device)
        dark_frame_v = beta * (alpha3 * frame_v_ori) ** gamma
        frame_hsv[:, 2, :, :] = dark_frame_v
        frame_rgb3 = torch.clamp((hsv2rgb_torch(frame_hsv)- 0.5) * 2.0, -1.0, 1.0) 
    
        save_image(denorm(frame_rgb3), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "input_video_generate33.png"))
        
        alpha4 = alpha_values[4]
        beta = torch.tensor(1.0).to(im.device)
        gamma = torch.tensor(1.0).to(im.device)
        dark_frame_v = beta * (alpha4 * frame_v_ori) ** gamma
        frame_hsv[:, 2, :, :] = dark_frame_v
        frame_rgb4 = torch.clamp((hsv2rgb_torch(frame_hsv)- 0.5) * 2.0, -1.0, 1.0) 
    
        save_image(denorm(frame_rgb4), os.path.join("/home/zly/code/ECCV_2024/Unfolding_progressive_single_stage/visulization/", "input_video_generate44.png"))
        
        synthetic_reference = torch.cat((frame_rgb0, frame_rgb1, frame_rgb2, frame_rgb3, frame_rgb4), dim=1) 
        
        return synthetic_reference.view(B, number_refs, C, H, W)       

    """define some functions"""
    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.args.g_conv_dim, self.args.g_norm_fun, self.args.g_act_fun, self.args.g_use_sn).to(self.device)
        self.G_second = Generator(self.args.g_conv_dim, self.args.g_norm_fun, self.args.g_act_fun, self.args.g_use_sn).to(self.device)
        self.D = Discriminator(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
        
        self.T = InterMerge(self.args.t_conv_dim).to(self.device)
        self.T_fusion = InterMerge(self.args.t_conv_dim).to(self.device)
        
        self.T_second = InterMerge(self.args.t_conv_dim).to(self.device)
        self.T_fusion_second = InterMerge(self.args.t_conv_dim).to(self.device)
        
        self.R = RAFT(self.args).to(self.device)
        
        if self.args.parallel:
            self.G.to(self.args.gpu_ids[0])
            self.G_second.to(self.args.gpu_ids[0])
            self.D.to(self.args.gpu_ids[0])
            self.T.to(self.args.gpu_ids[0])
            self.T_fusion.to(self.args.gpu_ids[0])
            
            self.T_second.to(self.args.gpu_ids[0])
            self.T_fusion_second.to(self.args.gpu_ids[0])
            
            self.R.to(self.args.gpu_ids[0])
            
            self.G = nn.DataParallel(self.G, self.args.gpu_ids)
            self.G_second = nn.DataParallel(self.G_second, self.args.gpu_ids)
            self.D = nn.DataParallel(self.D, self.args.gpu_ids)
            self.T = nn.DataParallel(self.T, self.args.gpu_ids)
            self.T_fusion = nn.DataParallel(self.T_fusion, self.args.gpu_ids)
            
            self.T_second = nn.DataParallel(self.T_second, self.args.gpu_ids)
            self.T_fusion_second = nn.DataParallel(self.T_fusion_second, self.args.gpu_ids)
            
            
            self.R = nn.DataParallel(self.R, self.args.gpu_ids)
        print("=== Models have been created ===")
        
        # print network
        if self.args.is_print_network:
            self.print_network(self.G, 'Generator')
            self.print_network(self.G_second, 'Generator_second')
            self.print_network(self.D, 'Discriminator')
            
            self.print_network(self.T, 'InterMerge')
            self.print_network(self.T_fusion, 'InterMerge_fusion')
            
            self.print_network(self.T_second, 'InterMerge second')
            self.print_network(self.T_fusion_second, 'InterMerge_fusion second')
            
            self.print_network(self.R, 'Raft')


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
            self.T.load_state_dict(checkpoint['T_net'])
            self.T_fusion.load_state_dict(checkpoint['T_fusion_net'])
            
            self.T_second.load_state_dict(checkpoint['T_second_net'])
            self.T_fusion_second.load_state_dict(checkpoint['T_fusion_second_net'])
            
            self.R.load_state_dict(checkpoint['R_net'])
            
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

        print("=========== loaded trained models (epochs: {})! ===========".format(resume_epochs))


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_path)