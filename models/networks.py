import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import math
import numpy as np

###############################################################################
# Helper Functions
###############################################################################

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'ed':
        # same downsample net as in ResnetGenerator, but adding more blocks in the beginning in upsample net than in ResnetGenerator.
        # in total, it has 2*n_blocks blocks
        netG = EDGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[], feat_matching=False):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, feat_matching=feat_matching)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, feat_matching=feat_matching)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)


def define_M(input_nc, output_nc, ngf, which_model_netM, norm='batch', use_dropout=False, init_type='normal',init_gain=0.02, gpu_ids=[], add_position_signal=False):
    norm_layer = get_norm_layer(norm_type=norm)
    netM = MaskPredictor(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=2, add_position_signal=add_position_signal)
    return init_net(netM, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
# TODO (yi 2020): The codes related to feat_matching is added in extra.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, feat_matching=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.n_layers = n_layers
        self.feat_matching = feat_matching
        self.use_sigmoid = use_sigmoid

        kw = 4
        padw = 1
        sequence = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if self.feat_matching:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)


    def forward(self, input):
        if self.feat_matching:
            res = [input]
            if self.use_sigmoid:
                length = self.n_layers + 3
            else:
                length = self.n_layers + 2
            for n in range(length):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class CoordConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        self.with_r = with_r
        if self.with_r:
            super(CoordConv2d, self).__init__(in_channels+3, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        else:
            super(CoordConv2d, self).__init__(in_channels+2, out_channels, kernel_size, stride, padding, dilation, groups,
                                         bias)

    def forward(self, input):
        batch_dim, _, y_dim, x_dim = list(input.size())

        xx_ones = torch.ones(batch_dim, y_dim) # x_dim in coord paper, but I think it's y_dim instead
        xx_ones = torch.unsqueeze(xx_ones, -1) # [batch_dim, y_dim, 1]

        xx_range= torch.unsqueeze(torch.arange(x_dim), 0) # [1, x_dim]
        xx_range= xx_range.expand(batch_dim, -1) # [batch_dim, x_dim]
        xx_range =torch.unsqueeze(xx_range, 1) # [batch_dim, 1, x_dim]

        xx_channel = torch.matmul(xx_ones, xx_range.type(torch.FloatTensor)) # [batch, y_dim, x_dim]
        xx_channel = torch.unsqueeze(xx_channel, 1) # [batch, 1, y_dim, x_dim]

        yy_ones = torch.ones(batch_dim, x_dim) # y_dim in coord paper, but I think it's x_dim instead
        yy_ones = torch.unsqueeze(yy_ones, 1) # [batch, 1, x_dim]

        yy_range = torch.unsqueeze(torch.arange(y_dim), 0) # [1, y_dim]
        yy_range = yy_range.expand(batch_dim, -1)  # [batch_dim, y_dim]
        yy_range = torch.unsqueeze(yy_range, -1)  # [batch_dim, y_dim, 1]

        yy_channel = torch.matmul(yy_range.type(torch.FloatTensor), yy_ones)  # [batch, y_dim, x_dim]
        yy_channel = torch.unsqueeze(yy_channel, 1)  # [batch, 1, y_dim, x_dim]

        xx_channel = xx_channel/(x_dim-1) * 2 -1
        yy_channel = yy_channel/(y_dim-1) * 2 -1

        ret = torch.cat((input, xx_channel.cuda(), yy_channel.cuda()), 1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel-0.5, 2) + torch.pow(yy_channel-0.5, 2))
            ret = torch.cat((ret, rr), 1)
            return F.conv1d(ret, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        return F.conv1d(ret, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class EDGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
            use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(EDGenerator, self).__init__()
        self.g_down = self._downsample_stream(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type, n_downsampling=2)
        self.g_up = self._upsample_stream(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type, n_upsampling=2)

    def _downsample_stream(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', n_downsampling=2):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            self.mult = 2**i
            model += [nn.Conv2d(ngf * self.mult, ngf * self.mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * self.mult * 2),
                      nn.ReLU(True)]

        self.mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * self.mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model = nn.Sequential(*model)
        return model

    def _upsample_stream(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', n_upsampling=2, use_tanh=True, use_sigmoid=False):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        mult = 2**n_upsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_upsampling):
            mult = 2**(n_upsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        if use_tanh:
            model += [nn.Tanh()]
        if use_sigmoid:
            model += [nn.Sigmoid()]
        model = nn.Sequential(*model)
        return model

    def forward(self, input_img):
        g_down = self.g_down(input_img)
        g_up = self.g_up(g_down)
        return g_up


class MaskPredictor(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
            use_dropout=False, n_blocks=2, padding_type='reflect', use_gt_mask=False,
            duo_att_ratio=1, add_position_signal=False):
        super(MaskPredictor, self).__init__()

        self.use_gt_mask = use_gt_mask
        self.add_position_signal = add_position_signal
        self.duo_att_ratio = duo_att_ratio

        if not self.use_gt_mask:
            self.real_stream, self.out_dim = self._downsample_stream_gate(input_nc, output_nc, ngf, norm_layer,
                                                                          use_dropout, n_blocks=n_blocks, padding_type=padding_type, n_downsampling=4)
            self.fake_stream, self.out_dim = self._downsample_stream_gate(input_nc, output_nc, ngf, norm_layer,
                                                                          use_dropout, n_blocks=n_blocks, padding_type=padding_type, n_downsampling=4)
            self.out_gated_stream = self._upsample_stream_gate(input_nc, output_nc, ngf, norm_layer,
                                                                          use_dropout, padding_type=padding_type, n_upsampling=4,
                                                                          use_sigmoid=True, use_tanh=False)
            self._add_duo_att(self.out_dim, self.out_dim // self.duo_att_ratio, norm_layer)

    def _add_duo_att(self, d1, d2, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # B d_l H W
        left_conv1_stream = [
            nn.Conv2d(
                d1, d2, kernel_size=1, stride=1, padding=0, bias=use_bias
            ),
            norm_layer(d2),
        ]
        self.left_conv1_stream = nn.Sequential(*left_conv1_stream)

        # B d_l H W
        left_conv2_stream = [
            nn.Conv2d(
                d1, d2, kernel_size=1, stride=1, padding=0, bias=use_bias
            ),
            norm_layer(d2),
            nn.ReLU(inplace=True),
        ]
        self.left_conv2_stream = nn.Sequential(*left_conv2_stream)

        # B H W d_l
        right_conv1_stream = [
            # B d_r H W
            nn.Conv2d(
                d1, d2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(d2),
        ]
        self.right_conv1_stream = nn.Sequential(*right_conv1_stream)

        # B d_r H W
        right_conv2_stream = [
            nn.Conv2d(
                d1, d2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            # B d_r H W
            norm_layer(d2),
            nn.ReLU(inplace=True),
        ]
        self.right_conv2_stream = nn.Sequential(*right_conv2_stream)

    def _downsample_stream_gate(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=2, padding_type='reflect', n_downsampling=4):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=2,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            self.mult = 1
            model += [nn.Conv2d(ngf, ngf, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf),
                      nn.ReLU(True)]

        self.mult = 1
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * self.mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model += [nn.ReLU(True)]

        model = nn.Sequential(*model)
        return model, ngf * self.mult

    def _upsample_stream_gate(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect', n_upsampling=4, use_tanh=True, use_sigmoid=False):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        model += [nn.Upsample(scale_factor=(n_upsampling ** 2 * 2 * 1), mode='nearest')]

        model = nn.Sequential(*model)
        return model

    def _position_signal_nd_numpy(self, tensor_size, min_timescale=1.0, max_timescale=1.0e4):
        # tensor_size = [batch, channel, d_1, d_2,.., d_n], in image case n=2
        num_dims = len(tensor_size)-2 # minus batch dim and channel dim
        channels = tensor_size[1]
        num_timescales = channels // (num_dims * 2)
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (num_timescales - 1))
        inv_timescales = min_timescale * np.exp(np.arange(num_timescales * 1.0) * - log_timescale_increment)
        x = np.zeros(tuple(tensor_size)).astype(np.float32)
        for dim in range(num_dims):
            dim_size = tensor_size[dim+2]
            position = np.arange(dim_size * 1.0)
            scaled_time = np.matmul(inv_timescales[:, np.newaxis], position[np.newaxis, :])
            signal = np.concatenate((np.sin(scaled_time), np.cos(scaled_time)), axis=0)
            prepad = dim * 2 * num_timescales
            postpad = channels - (dim + 1) * 2 * num_timescales
            signal = np.pad(signal, ((prepad, postpad), (0, 0)), 'constant')
            signal = np.expand_dims(signal, 0)
            for _ in range(1 + dim -1):
                signal = np.expand_dims(signal, -2)
            for _ in range(num_dims -1 -dim):
                signal = np.expand_dims(signal, -1)
            x += signal
        return x

    def _duo_forward(self, l, r, d2, h, w):
        # -----------------------------------
        # B d_l d_r
        self.left_conv1 = self.left_conv1_stream(l)
        self.right_conv1 = self.right_conv1_stream(r)
        self.mul = torch.matmul(
            self.left_conv1.view(1, d2, -1),
            self.right_conv1.view(1, d2, -1).permute(0, 2, 1).contiguous()
        )

        # -----------------------------------
        # B d_l d_r
        # B d_l d_r
        self.norm_mul = torch.sqrt(F.relu(self.mul)) - torch.sqrt(F.relu(-self.mul))
        self.norm_mul = torch.mul(self.norm_mul, 1.0 / d2)

        # B d_r d_l
        self.left_softmax = nn.Softmax(dim=2)(self.norm_mul.permute(0, 2, 1).contiguous())
        # B d_l d_r
        self.right_softmax = nn.Softmax(dim=2)(self.norm_mul)

        # -----------------------------------
        # B d_l HxW
        self.left_conv2 = self.left_conv2_stream(l).reshape(1, d2, -1)
        # B d_r HxW
        self.left_out = torch.matmul(
            self.left_softmax,
            self.left_conv2
        )
        self.left_out = l.add(self.left_out.view(1, d2, h, w))
        self.left_out = nn.ReLU(inplace=True)(self.left_out)

        # B d_r HxW
        self.right_conv2 = self.right_conv2_stream(r).reshape(1, d2, -1)
        # B d_l HxW
        self.right_out = torch.matmul(
            self.right_softmax,
            self.right_conv2
        )
        self.right_out = r.add(self.right_out.view(1, d2, h, w))
        self.right_out = nn.ReLU(inplace=True)(self.right_out)
        return self.left_out, self.right_out

    def forward(self, input_img, ground_truth, gt_mask=None):
        if self.use_gt_mask:
            gate_out = gt_mask.float()
        else:
            gate_real_mid = self.real_stream.forward(input_img)
            gate_fake_mid = self.fake_stream.forward(ground_truth)
            if self.add_position_signal: # add positions (todo): could also try using different position with different flags.
                gate_real_mid = gate_real_mid + torch.from_numpy(self._position_signal_nd_numpy(list(gate_real_mid.size()), 1, 128))
                gate_fake_mid = gate_fake_mid + torch.from_numpy(self._position_signal_nd_numpy(list(gate_fake_mid.size()), 1, 128))

        gate_real_mid, gate_fake_mid = self._duo_forward(
            gate_real_mid, gate_fake_mid, self.out_dim // self.duo_att_ratio, 8, 8
        )
        gate_mid = torch.nn.CosineSimilarity().forward(gate_real_mid, gate_fake_mid).unsqueeze(1)
        gate_out = self.out_gated_stream.forward(gate_mid)
        gate_sum = gate_mid.sum(3).sum(2)

        return gate_out, gate_sum


##############################################################################
# Out-of-Date
# A GatedGenerator actually can be splitted into a Generator and a MaskPredictor
##############################################################################


class GatedGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
            use_dropout=False, n_blocks=6, padding_type='reflect', use_gt_mask=False,
            duo_att_ratio=1, add_position_signal=False):
        super(GatedGenerator, self).__init__()
        self.use_gt_mask = use_gt_mask
        self.add_position_signal = add_position_signal
        if not self.use_gt_mask:
            self.real_stream, self.out_dim = self._downsample_stream_gate(input_nc, output_nc)
            self.fake_stream, self.out_dim = self._downsample_stream_gate(input_nc, output_nc)
            self.out_gated_stream = self._upsample_stream_gate(input_nc, output_nc, use_sigmoid=True, use_tanh=False)
            self.duo_att_ratio = duo_att_ratio
            self._add_duo_att(self.out_dim, self.out_dim // self.duo_att_ratio, norm_layer)

        self.g_down = self._downsample_stream(input_nc, output_nc)
        self.g_up = self._upsample_stream(input_nc, output_nc)

    def _add_duo_att(self, d1, d2, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # B d_l H W
        left_conv1_stream = [
            nn.Conv2d(
                d1, d2, kernel_size=1, stride=1, padding=0, bias=use_bias
            ),
            norm_layer(d2),
        ]
        self.left_conv1_stream = nn.Sequential(*left_conv1_stream)

        # B d_l H W
        left_conv2_stream = [
            nn.Conv2d(
                d1, d2, kernel_size=1, stride=1, padding=0, bias=use_bias
            ),
            norm_layer(d2),
            nn.ReLU(inplace=True),
        ]
        self.left_conv2_stream = nn.Sequential(*left_conv2_stream)

        # B H W d_l
        right_conv1_stream = [
            # B d_r H W
            nn.Conv2d(
                d1, d2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(d2),
        ]
        self.right_conv1_stream = nn.Sequential(*right_conv1_stream)

        # B d_r H W
        right_conv2_stream = [
            nn.Conv2d(
                d1, d2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            # B d_r H W
            norm_layer(d2),
            nn.ReLU(inplace=True),
        ]
        self.right_conv2_stream = nn.Sequential(*right_conv2_stream)

    def _downsample_stream(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', n_downsampling=2):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            self.mult = 2**i
            model += [nn.Conv2d(ngf * self.mult, ngf * self.mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * self.mult * 2),
                      nn.ReLU(True)]

        self.mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * self.mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model = nn.Sequential(*model)
        return model

    def _upsample_stream(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', n_upsampling=2, use_tanh=True, use_sigmoid=False):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        mult = 2**n_upsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_upsampling):
            mult = 2**(n_upsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        if use_tanh:
            model += [nn.Tanh()]
        if use_sigmoid:
            model += [nn.Sigmoid()]
        model = nn.Sequential(*model)
        return model

    def _downsample_stream_gate(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=2, padding_type='reflect', n_downsampling=4):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=2,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            self.mult = 1
            model += [nn.Conv2d(ngf, ngf, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf),
                      nn.ReLU(True)]

        self.mult = 1
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * self.mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model += [nn.ReLU(True)]

        model = nn.Sequential(*model)
        return model, ngf * self.mult

    def _upsample_stream_gate(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect', n_upsampling=4, use_tanh=True, use_sigmoid=False):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        model += [nn.Upsample(scale_factor=(n_upsampling ** 2 * 2 * 1), mode='nearest')]

        model = nn.Sequential(*model)
        return model

    def forward(self, input_img, ground_truth, gt_mask=None):
        if self.use_gt_mask:
            gate_out = gt_mask.float()
        else:
            gate_real_mid = self.real_stream.forward(input_img)
            gate_fake_mid = self.fake_stream.forward(ground_truth)
            if self.add_position_signal:
                gate_real_mid = gate_real_mid + torch.from_numpy(self._position_signal_nd_numpy(list(gate_real_mid.size()), 1, 128)).to(gate_real_mid.device)
                gate_fake_mid = gate_fake_mid + torch.from_numpy(self._position_signal_nd_numpy(list(gate_fake_mid.size()), 1, 128)).to(gate_fake_mid.device)

            gate_real_mid, gate_fake_mid = self._duo_forward(
                gate_real_mid, gate_fake_mid, self.out_dim // self.duo_att_ratio, 8, 8
            )

            gate_mid = torch.nn.CosineSimilarity().forward(
                    gate_real_mid, gate_fake_mid,
                ).unsqueeze(1)
            gate_out = self.out_gated_stream.forward(gate_mid)

        g_down = self.g_down(input_img)
        g_up = self.g_up(g_down)
        gate_sum = gate_mid.sum(3).sum(2)

        return g_up, gate_out, gate_sum



    def _position_signal_nd_numpy(self, tensor_size, min_timescale=1.0, max_timescale=1.0e4):
        # tensor_size = [batch, channel, d_1, d_2,.., d_n], in image case n=2
        num_dims = len(tensor_size)-2 # minus batch dim and channel dim
        channels = tensor_size[1]
        num_timescales = channels // (num_dims * 2)
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (num_timescales - 1))
        inv_timescales = min_timescale * np.exp(np.arange(num_timescales * 1.0) * - log_timescale_increment)
        x = np.zeros(tuple(tensor_size)).astype(np.float32)
        for dim in range(num_dims):
            dim_size = tensor_size[dim+2]
            position = np.arange(dim_size * 1.0)
            scaled_time = np.matmul(inv_timescales[:, np.newaxis], position[np.newaxis, :])
            signal = np.concatenate((np.sin(scaled_time), np.cos(scaled_time)), axis=0)
            prepad = dim * 2 * num_timescales
            postpad = channels - (dim + 1) * 2 * num_timescales
            signal = np.pad(signal, ((prepad, postpad), (0, 0)), 'constant')
            signal = np.expand_dims(signal, 0)
            for _ in range(1 + dim -1):
                signal = np.expand_dims(signal, -2)
            for _ in range(num_dims -1 -dim):
                signal = np.expand_dims(signal, -1)
            x += signal
        return x

    def _duo_forward(self, l, r, d2, h, w):
        # -----------------------------------
        # B d_l d_r
        self.left_conv1 = self.left_conv1_stream(l)
        self.right_conv1 = self.right_conv1_stream(r)
        self.mul = torch.matmul(
            self.left_conv1.view(1, d2, -1),
            self.right_conv1.view(1, d2, -1).permute(0, 2, 1).contiguous()
        )

        # -----------------------------------
        # B d_l d_r
        # B d_l d_r
        self.norm_mul = torch.sqrt(F.relu(self.mul)) - torch.sqrt(F.relu(-self.mul))
        self.norm_mul = torch.mul(self.norm_mul, 1.0 / d2)

        # B d_r d_l
        self.left_softmax = nn.Softmax(dim=2)(self.norm_mul.permute(0, 2, 1).contiguous())
        # B d_l d_r
        self.right_softmax = nn.Softmax(dim=2)(self.norm_mul)

        # -----------------------------------
        # B d_l HxW
        self.left_conv2 = self.left_conv2_stream(l).reshape(1, d2, -1)
        # B d_r HxW
        self.left_out = torch.matmul(
            self.left_softmax,
            self.left_conv2
        )
        self.left_out = l.add(self.left_out.view(1, d2, h, w))
        self.left_out = nn.ReLU(inplace=True)(self.left_out)

        # B d_r HxW
        self.right_conv2 = self.right_conv2_stream(r).reshape(1, d2, -1)
        # B d_l HxW
        self.right_out = torch.matmul(
            self.right_softmax,
            self.right_conv2
        )
        self.right_out = r.add(self.right_out.view(1, d2, h, w))
        self.right_out = nn.ReLU(inplace=True)(self.right_out)
        return self.left_out, self.right_out


def define_gated_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal',init_gain=0.02, gpu_ids=[], use_gt_mask=False, add_position_signal=False):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    netG = GatedGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, use_gt_mask=use_gt_mask, add_position_signal=add_position_signal)
    return init_net(netG, init_type, init_gain, gpu_ids)
