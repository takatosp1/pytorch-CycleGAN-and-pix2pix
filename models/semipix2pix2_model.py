import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import numpy as np
sign = networks.LBSign.apply

class SemiPix2Pix2Model(BaseModel):
    def name(self):
        return 'SemiPix2Pix2Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch', gt_crop=1)
        parser.set_defaults(dataset_mode='semialigned')
        parser.set_defaults(which_model_netG='unet_256')
        # if is_train:
        #     parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_GAN_1', 'G_GAN_2', 'G_L1', 'D_real', 'D_real_1', 'D_real_2', 'D_fake', 'D_fake_1', 'D_fake_2']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'pred_gate', \
            'real_A_w_pred_gate', 'real_B_w_pred_gate', 'fake_B_w_pred_gate', \
                'real_A_in_pred_gate', 'real_B_in_pred_gate', 'fake_B_in_pred_gate']
        if self.isTrain:
            self.visual_names.append('real_gate')
            self.visual_names.append('real_B_w_real_gate')
            self.visual_names.append('fake_B_w_real_gate')
            # self.visual_names.append('pred_real')
            # self.visual_names.append('pred_fake')
            # self.visual_names.append('D')
        if self.opt.visualize_L1_loss:
            self.visual_names.append('L1')
            self.visual_names.append('gated_L1')

        if not self.isTrain:
            self.visual_names.append('real_gate')

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_gated_G(opt.input_nc, opt.output_nc, opt.ngf,
                                            opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
                                            self.opt.use_gt_mask, self.opt.add_position_signal)

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
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_gate = input['gate'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B, self.pred_gate, self.gate_sum = self.netG(self.real_A, self.real_B, self.real_gate)
        if not self.isTrain:
            # enable straight through estimator (if implemented correctly) in training predicts worse mask
            self.pred_gate = (sign(self.pred_gate - 0.5) + 1 ) / 2.0

        self.fake_B_w_pred_gate = self.fake_B * self.pred_gate
        self.real_B_w_pred_gate = self.real_B * self.pred_gate
        self.real_A_w_pred_gate = self.real_A * self.pred_gate

        self.fake_B_in_pred_gate = self.fake_B * (1-self.pred_gate)
        self.real_B_in_pred_gate = self.real_B * (1-self.pred_gate)
        self.real_A_in_pred_gate = self.real_A * (1-self.pred_gate)

        # For visualize
        self.gated_L1 = (self.fake_B_w_pred_gate - self.real_B_w_pred_gate).clone().sum(1).unsqueeze(dim=1)
        self.L1 = (self.fake_B - self.real_B).clone().sum(1).unsqueeze(dim=1)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB_1 = self.fake_AB_pool.query(
            torch.cat((self.real_A_w_pred_gate, self.fake_B_w_pred_gate), 1))  # TODO? real_A_w_pred_gate vs fake_B_w_pred_gate
        fake_AB_2 = self.fake_AB_pool.query(
            torch.cat((self.real_A_in_pred_gate, self.fake_B_in_pred_gate), 1))
        pred_fake_1 = self.netD(fake_AB_1.detach())
        pred_fake_2 = self.netD(fake_AB_2.detach())
        self.loss_D_fake_1 = self.criterionGAN(pred_fake_1, False)
        self.loss_D_fake_2 = self.criterionGAN(pred_fake_2, False)
        self.loss_D_fake = (self.loss_D_fake_1 + self.loss_D_fake_2) * 0.5

        # Real
        real_AB_1 = torch.cat((self.real_A_w_pred_gate, self.real_B_w_pred_gate), 1) # TODO? real_A_w_pred_gate vs real_B_w_pred_gate
        real_AB_2 = torch.cat((self.real_A_in_pred_gate, self.real_B_in_pred_gate), 1)
        pred_real_1 = self.netD(real_AB_1.detach())
        pred_real_2 = self.netD(real_AB_2.detach())
        self.loss_D_real_1 = self.criterionGAN(pred_real_1, True)
        self.loss_D_real_2 = self.criterionGAN(pred_real_2, True)
        self.loss_D_real = (self.loss_D_real_1 + self.loss_D_real_2) * 0.5

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        # For Visualization
        self.real_B_w_real_gate = self.real_B * self.real_gate
        self.fake_B_w_real_gate = self.fake_B * self.real_gate
        self.pred_fake_1 = 1 - pred_fake_1.clone()
        self.pred_fake_2 = 1 - pred_fake_2.clone()
        self.pred_fake = (self.pred_fake_1 + self.pred_fake_2) * 0.5
        self.pred_real = (pred_real_1.clone() + pred_real_2.clone()) * 0.5
        self.D = self.pred_fake + self.pred_real

        if self.opt.use_area_constraint:
            mask_target = torch.tensor([[2 * 2 * 0.25]]).to(self.device)
            self.loss_mask = self.criterionMask(self.gate_sum, mask_target)
            self.loss_mask *= 1000000
            self.loss_mask.backward(retain_graph=True, create_graph=True)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB_1 = self.fake_AB_pool.query(
            torch.cat((self.real_A_w_pred_gate, self.fake_B_w_pred_gate), 1))  # TODO? real_A_w_pred_gate vs fake_B_w_pred_gate
        fake_AB_2 = self.fake_AB_pool.query(
            torch.cat((self.real_A_in_pred_gate, self.fake_B_in_pred_gate), 1))
        pred_fake_1 = self.netD(fake_AB_1)
        pred_fake_2 = self.netD(fake_AB_2)
        self.loss_G_GAN_1 = self.criterionGAN(pred_fake_1, True)
        self.loss_G_GAN_2 = self.criterionGAN(pred_fake_2, True)
        self.loss_G_GAN = (self.loss_G_GAN_1 + self.loss_G_GAN_2) * 0.5

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B_w_pred_gate, self.real_B) * self.opt.lambda_L1

        # loss for pred_gate
        # self.real_A_w_pred_gate = self.real_A * self.pred_gate.detach()
        # real_AB = torch.cat((self.real_A_w_pred_gate, self.real_B_w_pred_gate), 1)
        # pred_real = self.netD(real_AB)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
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
