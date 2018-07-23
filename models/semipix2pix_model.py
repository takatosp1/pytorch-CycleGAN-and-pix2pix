import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class SemiPix2PixModel(BaseModel):
    def name(self):
        return 'SemiPix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'gate']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_gated_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.criterionMask = nn.MSELoss()

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = []
        self.real_gate = []
        for stream_num in range(self.opt.num_stream):
            self.real_B.append(
                input['B' if AtoB else 'A'][stream_num].clone().to(self.device))
            if self.opt.use_gt_mask:
                self.real_gate.append(input[""][stream_num].clone().to(self.device))
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = []
        self.gate = []
        self.sum_gate = []
        for stream_num in range(self.opt.num_stream):
            if self.opt.add_constraint:
                fake_B, gate, sum_gate = \
                    self.netG(self.real_A, self.real_B[stream_num], self.real_gate[stream_num], constraint=True)
            else:
                fake_B, gate = \
                    self.netG(self.real_A, self.real_B[stream_num], self.real_gate[stream_num], constraint=False)
            self.fake_B.append(fake_B.clone())
            self.gate.append(gate.clone())
            if self.opt.add_constraint:
                self.sum_gate.append(sum_gate.clone())
        # TODO do we need a mask on the GT
        # self.fake_B, self.gate, self.gated_B = self.netG(self.real_A, self.real_B)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        for stream_num in range(self.opt.num_stream):
            fake_AB = self.fake_AB_pool.query(
                torch.cat((self.real_A, self.fake_B[stream_num]), 1))
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)

            if self.opt.add_constraint:
                mask_target = torch.tensor([[16 * 16 * 0.25]]).to(self.device)
                self.loss_mask = self.criterionMask(self.sum_gate[stream_num], mask_target)

            # Real
            # real_AB = torch.cat((self.real_A, self.gated_B), 1)
            # TODO do we need a mask on the GT
            real_AB = torch.cat((self.real_A, self.real_B[stream_num]), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)

            # Combined loss
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

            self.loss_D.backward(retain_graph=True, create_graph=True)

            if self.opt.add_constraint:
                self.loss_mask.backward(retain_graph=True, create_graph=True)


    def backward_G(self):
        # First, G(A) should fake the discriminator
        for stream_num in range(self.opt.num_stream):
            fake_AB = torch.cat((self.real_A, self.fake_B[stream_num]), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

            # Second, G(A) = B
            # self.loss_G_L1 = self.criterionL1(self.fake_B, self.gated_B) * self.opt.lambda_L1
            self.loss_G_L1 = self.criterionL1(
                self.fake_B[stream_num], self.real_B[stream_num]) * self.opt.lambda_L1

            # TODO do we need a mask on the GT
            # self.loss_G_L1 = torch.mean(torch.abs(self.fake_B - self.gated_B) * self.opt.lambda_L1)

            self.loss_G = self.loss_G_GAN + self.loss_G_L1

            self.loss_G.backward(retain_graph=True, create_graph=True)

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
