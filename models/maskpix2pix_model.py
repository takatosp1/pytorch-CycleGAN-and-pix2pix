import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import numpy as np
from itertools import chain
sign = networks.LBSign.apply



class MaskPix2PixModel(BaseModel):
    def name(self):
        return 'MaskPix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch', gt_crop=1)
        parser.set_defaults(dataset_mode='semialigned')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        if self.opt.backward_mode == "3step":
            self.loss_names.append('M_GAN')
            self.loss_names.append('M_L1')
        # TODO(yi): to experiment on the mask loss
        if self.opt.add_mask_L2_loss:
            self.loss_names.append("M_L2")

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'pred_mask', 'real_B_w_pred_mask', 'fake_B_w_pred_mask']
        if self.isTrain:
            self.visual_names.append('real_mask')
            self.visual_names.append('real_B_w_real_mask')
            self.visual_names.append('fake_B_w_real_mask')
            # self.visual_names.append('pred_real')
            # self.visual_names.append('pred_fake')
            # self.visual_names.append('D')
        if self.opt.visualize_L1_loss:
            self.visual_names.append('L1')
            self.visual_names.append('maskd_L1')

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D', 'M']
        else:  # during test time, only load Gs
            self.model_names = ['G', 'M']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netM = networks.define_M(opt.input_nc, opt.output_nc, opt.ngf, None,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      opt.add_position_signal)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # additional loss function for predicted mask
            self.criterionMask = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_M = torch.optim.Adam(self.netM.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_M)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_mask = input['gate'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        self.pred_mask, self.mask_sum = self.netM(self.real_A, self.real_B, self.real_mask)
        if not self.isTrain:
            # enable straight through estimator (if implemented correctly) in training predicts worse mask
            self.pred_mask = (sign(self.pred_mask - 0.5) + 1 ) / 2.0

        self.fake_B_w_pred_mask = self.fake_B * self.pred_mask
        self.real_B_w_pred_mask = self.real_B * self.pred_mask

        # For visualize
        self.masked_L1 = (self.fake_B_w_pred_mask - self.real_B_w_pred_mask).clone().sum(1).unsqueeze(dim=1)
        self.L1 = (self.fake_B - self.real_B).clone().sum(1).unsqueeze(dim=1)

        # if self.opt.add_mask_L2_loss:
        #     self.pred_mask_to_comp, self.mask_sum_to_comp = self.netM(self.real_A, self.real_B_w_pred_mask)

    def backward_D(self):
        fake_B = self.fake_B_w_pred_mask
        real_B = self.real_B

        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        # For Visualization
        self.real_B_w_real_mask = self.real_B * self.real_mask
        self.fake_B_w_real_mask = self.fake_B * self.real_mask
        self.pred_fake = 1 - pred_fake.clone()
        self.pred_real = pred_real.clone()
        self.D = self.pred_fake + self.pred_real

    def backward_GM(self):
        fake_B = self.fake_B_w_pred_mask
        real_B = self.real_B

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(fake_B, real_B) * self.opt.lambda_L1

        # Third, M(AB) = mask
        if self.opt.add_mask_L2_loss:
            self.pred_mask_to_comp, _ = self.netM(self.real_A, self.fake_B.detach() * self.pred_mask)
            # self.loss_M_L2 = torch.sum((self.pred_mask - pred_mask_to_comp) ** 2) / self.pred_mask.data.nelement() * self.opt.lambda_mask_L2
            self.loss_M_L2 = self.criterionMask(self.pred_mask, self.pred_mask_to_comp) * self.opt.lambda_mask_L2

        # if self.opt.use_area_constraint:
        #     mask_target = torch.tensor([[2 * 2 * 0.25]]).to(self.device)
        #     self.loss_mask = self.criterionMask(self.mask_sum, mask_target)
        #     self.loss_mask *= 1000000
        #     self.loss_mask.backward(retain_graph=True, create_graph=True)

        # Combined loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if self.opt.add_mask_L2_loss:
            self.loss_G += self.loss_M_L2
        self.loss_G.backward()

    def backward_G(self):
        mask_deatch = self.pred_mask.detach()
        fake_B = self.fake_B * mask_deatch
        real_B = self.real_B

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(fake_B, real_B) * self.opt.lambda_L1

        # Combined loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward(retain_graph=True) #to retain G since is used by backward_M

    def backward_M(self):
        fake_B_detach = self.fake_B.detach()
        # fake_B = torch.autograd.Variable(fake_B_detach * self.pred_mask, requires_grad=True)
        fake_B = fake_B_detach * self.pred_mask
        real_B = self.real_B

        # TODO(yi) not sure if 1 and 2 are needed.
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_M_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_M_L1 = self.criterionL1(fake_B, real_B) * self.opt.lambda_L1

        # Third, M(AB) = mask
        if self.opt.add_mask_L2_loss:
            self.pred_mask_to_comp, _ = self.netM(self.real_A, fake_B)
            # self.loss_M_L2 = torch.sum((self.pred_mask - pred_mask_to_comp) ** 2) / self.pred_mask.data.nelement() * self.opt.lambda_mask_L2
            self.loss_M_L2 = self.criterionMask(self.pred_mask, self.pred_mask_to_comp) * self.opt.lambda_mask_L2

        self.loss_M = self.loss_M_GAN + self.loss_M_L1
        if self.opt.add_mask_L2_loss:
            self.loss_M += self.loss_M_L2
        self.loss_M.backward()

    def optimize_parameters(self):
        self.forward()
        if self.opt.backward_mode =='3step':
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            # update G
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            # update M
            self.set_requires_grad(self.netD, False)
            self.optimizer_M.zero_grad()
            self.backward_M()
            self.optimizer_M.step()
        elif self.opt.backward_mode =='2step':
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            # update G
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.optimizer_M.zero_grad()
            self.backward_GM()
            self.optimizer_G.step()
            self.optimizer_M.step()
        else:
            raise KeyError("No such backward_mode " + self.opt.backward_mode)
