import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import math
import numpy as np

###############################################################################
# Helper Functions
###############################################################################


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
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'multiscale':
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=False,
                                       num_D=3, getIntermFeat=False) # (todo) num_D & geIntermFeat option
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            i =0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                i+=1
                loss += self.loss(pred, target_tensor.cuda())
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor.cuda())


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


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
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


class Interpolate(nn.Module):
    def __init__(self, scale_factor, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, size=self.size, mode=self.mode, align_corners=False)
        return x


class GatedGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
            use_dropout=False, n_blocks=6, padding_type='reflect', use_gt_mask=False,
            duo_att_ratio=1, add_position_signal=True, which_net_mask='basic'):
        super(GatedGenerator, self).__init__()
        self.use_gt_mask = use_gt_mask
        self.add_position_signal = add_position_signal
        self.which_net_mask = which_net_mask
        if not self.use_gt_mask:
            # self.real_stream, self.out_dim = self._downsample_stream_gate(input_nc, output_nc)
            # self.fake_stream, self.out_dim = self._downsample_stream_gate(input_nc, output_nc)
            self.downsample_gate_model, self.out_dim = self._downsample_stream_gate(input_nc, output_nc)
            for n in range(len(self.downsample_gate_model)):
                setattr(self, 'downsample_gate_model' + str(n), nn.Sequential(*self.downsample_gate_model[n]))
            self.out_gated_stream = self._upsample_stream_gate(input_nc, output_nc, use_sigmoid=True, use_tanh=False)
            self.duo_att_ratio = duo_att_ratio
            self._add_duo_att(self.out_dim, self.out_dim // self.duo_att_ratio, norm_layer)
            if self.which_net_mask == 'multiscale1':
                model = []
                for i in range(4+1): # todo, option, 1 + n_downsample
                    model += [nn.ConvTranspose2d(self.out_dim, self.out_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                              nn.BatchNorm2d(self.out_dim),
                              nn.ReLU(True)]
                    setattr(self, 'upsample_gate_model'+str(i), nn.Sequential(*model))
                model = [nn.Conv2d(self.out_dim * (1+4+1), self.out_dim, kernel_size=1, stride=1), norm_layer(self.out_dim), nn.ReLU(True)]  # todo, option, n_downsample
                setattr(self, 'conv_af_upsample_gate', nn.Sequential(*model))
            elif self.which_net_mask == 'multiscale2':
                model =[nn.Conv2d((1 + 4 + 1), 1, kernel_size=1, stride=1), nn.Sigmoid()]
                setattr(self, 'conv_af_upsample_gate', nn.Sequential(*model)) # todo, option, n_downsample
            elif self.which_net_mask == 'multiscale3':
                model = [nn.Conv2d(self.out_dim * (1 + 4 + 1), self.out_dim, kernel_size=1, stride=1),
                         norm_layer(self.out_dim), nn.ReLU(True)]  # todo, option, n_downsample
                setattr(self, 'conv_af_downsample_gate', nn.Sequential(*model))
                self.pool_of_duo_downsample = nn.Sequential(*[nn.AvgPool2d(3, stride=2, padding=[1, 1])])
            elif self.which_net_mask == 'multiscale4':
                model = [nn.Conv2d((1 + 4 + 1), 1, kernel_size=1, stride=1), nn.Sigmoid()] # todo, option, n_downsample
                setattr(self, 'conv_af_downsample_gate', nn.Sequential(*model))
                self.pool_of_duo_downsample = nn.Sequential(*[nn.AvgPool2d(3, stride=2, padding=[1, 1])])
            elif self.which_net_mask == 'multiscale5':
                out_dim = (1+4+1) * self.out_dim # todo, option, n_downsample
                self._add_duo_att(out_dim, out_dim // self.duo_att_ratio, norm_layer)
                model = [nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1),
                         norm_layer(out_dim), nn.ReLU(True)]  # todo, option, n_downsample
                setattr(self, 'conv_af_downsample_gate', nn.Sequential(*model))
                self.pool_of_duo_downsample = nn.Sequential(*[nn.AvgPool2d(3, stride=2, padding=[1, 1])])

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

        model = [[nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=2,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]]

        for i in range(n_downsampling):
            self.mult = 1
            model += [[nn.Conv2d(ngf, ngf, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf),
                      nn.ReLU(True)]]

        self.mult = 1
        for i in range(n_blocks):
            model += [[ResnetBlock(ngf * self.mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]]

        model += [[nn.ReLU(True)]]
        # model = nn.Sequential(*model)
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
        model += [nn.Upsample(scale_factor=(2 ** n_upsampling * 2 * 1), mode='nearest')]

        model = nn.Sequential(*model)
        return model

    def forward(self, input_img, ground_truth, gt_mask=None, use_area_constraint=0):
        if self.use_gt_mask:
            gate_out = gt_mask.float()
        else:
            gate_real_mid_res = [input_img]
            for n in range(len(self.downsample_gate_model)):
                model = getattr(self, 'downsample_gate_model'+str(n))
                gate_real_mid_res.append(model.forward(gate_real_mid_res[-1]))
            gate_real_mid_res = gate_real_mid_res[1:]

            gate_fake_mid_res = [ground_truth]
            for n in range(len(self.downsample_gate_model)):
                model = getattr(self, 'downsample_gate_model'+str(n))
                gate_fake_mid_res.append(model(gate_fake_mid_res[-1]))
            gate_fake_mid_res = gate_fake_mid_res[1:]

            if self.which_net_mask=='basic':
                gate_real_mid = gate_real_mid_res[-1]
                gate_fake_mid = gate_fake_mid_res[-1]
                if self.add_position_signal:  # add positions (todo): could also try using different position with different flags.
                    gate_real_mid = gate_real_mid + torch.from_numpy(
                        self._position_signal_nd_numpy(list(gate_real_mid.size()), 1, 128))
                    gate_fake_mid = gate_fake_mid + torch.from_numpy(
                        self._position_signal_nd_numpy(list(gate_fake_mid.size()), 1, 128))
                gate_real_mid, gate_fake_mid = self._duo_forward(
                    gate_real_mid, gate_fake_mid, self.out_dim // self.duo_att_ratio, 8, 8
                )
                gate_mid = torch.nn.CosineSimilarity().forward(
                    gate_real_mid, gate_fake_mid,
                ).unsqueeze(1)
                gate_out = self.out_gated_stream.forward(gate_mid)

            elif self.which_net_mask == 'multiscale1': # duo, upsample, concat, consine similarity
                gate_real_duo_res = []
                gate_fake_duo_res = []
                for i in range(1+4): # todo, option, 1 + n_downsample
                    gate_real_duo, gate_fake_duo = self._duo_forward(
                        gate_real_mid_res[i], gate_fake_mid_res[i], self.out_dim // self.duo_att_ratio, 8*(2**(4-i)), 8*(2**(4-i)) # todo, option, n_downsample
                    )
                    gate_real_duo_res.append(gate_real_duo)
                    gate_fake_duo_res.append(gate_fake_duo)
                gate_real_duo, gate_fake_duo = self._duo_forward(
                    gate_real_mid_res[-1], gate_fake_mid_res[-1], self.out_dim // self.duo_att_ratio, 8, 8)
                gate_real_duo_res.append(gate_real_duo)
                gate_fake_duo_res.append(gate_fake_duo)

                for i in range(4+1): # todo, option, 1 + n_downsample
                    seq = getattr(self, 'upsample_gate_model' + str(i))
                    gate_fake_duo_res[i], gate_real_duo_res[i] = seq.forward(gate_fake_duo_res[i]), seq.forward(gate_real_duo_res[i])
                gate_fake_duo_res[-1], gate_real_duo_res[-1] = seq.forward(gate_fake_duo_res[-1]), seq.forward(gate_real_duo_res[-1])

                # seq = getattr(self, 'conv_af_upsample_gate')
                # gate_real_mid = seq.forward(torch.cat(gate_real_duo_res, 1))
                # gate_fake_mid = seq.forward(torch.cat(gate_fake_duo_res, 1))
                gate_real_mid = torch.cat(gate_real_duo_res, 1)
                gate_fake_mid = torch.cat(gate_fake_duo_res, 1)
                gate_out = torch.nn.CosineSimilarity().forward(gate_real_mid, gate_fake_mid).unsqueeze(1)
            elif self.which_net_mask == 'multiscale2': # duo, consine sim, upsample, concat
                gate_out_res = []
                for i in range(1 + 4):  # todo, option, 1 + n_downsample
                    gate_real_duo, gate_fake_duo = self._duo_forward(
                            gate_real_mid_res[i], gate_fake_mid_res[i], self.out_dim // self.duo_att_ratio,
                            8 * (2 ** (4 - i)), 8 * (2 ** (4 - i))# todo, option, n_downsample
                    )
                    gate_mid = torch.nn.CosineSimilarity().forward(gate_real_duo, gate_fake_duo,).unsqueeze(1)
                    gate_out = nn.Sequential(*[nn.Upsample(scale_factor=(2 ** i * 2 * 1), mode='nearest'), ]).forward(gate_mid)
                    gate_out_res.append(gate_out)
                gate_real_duo, gate_fake_duo = self._duo_forward(
                    gate_real_mid_res[i], gate_fake_mid_res[i], self.out_dim // self.duo_att_ratio, 8, 8)
                gate_mid = torch.nn.CosineSimilarity().forward(gate_real_duo, gate_fake_duo, ).unsqueeze(1)
                gate_out = self.out_gated_stream.forward(gate_mid)
                gate_out_res.append(gate_out)

                gate_out = torch.cat(gate_out_res, 1)
                model = getattr(self, 'conv_af_upsample_gate')
                gate_out = model.forward(gate_out)
                # n, c, w, h = gate_out.size()
                # gate_out = gate_out.view(n, c, w * h).permute(0, 2, 1)
                # gate_out = F.max_pool1d(gate_out, 6, 1).permute(0, 2, 1).view(n, 1, w, h)
            elif self.which_net_mask == 'multiscale3': # duo, downsample, concat, cosine sim, (upsample)
                gate_real_duo_res = []
                gate_fake_duo_res = []
                for i in range(1 + 4):  # todo, option, 1 + n_downsample
                    gate_real_duo, gate_fake_duo = self._duo_forward(
                        gate_real_mid_res[i], gate_fake_mid_res[i], self.out_dim // self.duo_att_ratio,
                                                                    8 * (2 ** (4 - i)), 8 * (2 ** (4 - i)))  # todo, option, n_downsample
                    gate_real_duo_res.append(gate_real_duo)
                    gate_fake_duo_res.append(gate_fake_duo)
                gate_real_duo, gate_fake_duo = self._duo_forward(
                    gate_real_mid_res[-1], gate_fake_mid_res[-1], self.out_dim // self.duo_att_ratio, 8, 8)
                gate_real_duo_res.append(gate_real_duo)
                gate_fake_duo_res.append(gate_fake_duo)

                # downsample with orignal dowmsample_gate_model
                for n in range(1, 4+1): # todo, option, 1 + n_downsample
                    seq = getattr(self, 'downsample_gate_model' + str(n))
                    for i in range(n):
                        gate_fake_duo_res[i], gate_real_duo_res[i] = seq.forward(gate_fake_duo_res[i]), seq.forward(gate_real_duo_res[i])
                for n in range(4+1, len(self.downsample_gate_model)-1): # todo, option, 1 + n_downsample
                    seq = getattr(self, 'downsample_gate_model' + str(n))
                    for i in range(len(gate_fake_duo_res)-1):
                        gate_fake_duo_res[i], gate_real_duo_res[i] = seq.forward(gate_fake_duo_res[i]), seq.forward(
                            gate_real_duo_res[i])

                # # downsample with avgpool net
                # for n in range(4): # todo, option, n_downsample
                #     seq = self.pool_of_duo_downsample
                #     for i in range(n+1):
                #         gate_fake_duo_res[i], gate_real_duo_res[i] = seq.forward(gate_fake_duo_res[i]), seq.forward(
                #             gate_real_duo_res[i])

                # seq = getattr(self, 'conv_af_downsample_gate')
                # gate_real_mid = seq.forward(torch.cat(gate_real_duo_res, 1))
                # gate_fake_mid = seq.forward(torch.cat(gate_fake_duo_res, 1))
                gate_real_mid = torch.cat(gate_real_duo_res, 1)
                gate_fake_mid = torch.cat(gate_fake_duo_res, 1)
                gate_mid = torch.nn.CosineSimilarity().forward(gate_real_mid, gate_fake_mid).unsqueeze(1)
                gate_out = self.out_gated_stream.forward(gate_mid)
            elif self.which_net_mask == 'multiscale4': # duo, consine sim, downsample, concat, (upsample) (not make too much sense here)
                gate_mid_res = []
                for i in range(1 + 4):  # todo, option, 1 + n_downsample
                    gate_real_duo, gate_fake_duo = self._duo_forward(
                        gate_real_mid_res[i], gate_fake_mid_res[i], self.out_dim // self.duo_att_ratio,
                                                                    8 * (2 ** (4 - i)), 8 * (2 ** (4 - i))) # todo, option, n_downsample
                    gate_mid = torch.nn.CosineSimilarity().forward(gate_real_duo, gate_fake_duo,).unsqueeze(1)
                    gate_mid_res.append(gate_mid)
                gate_real_duo, gate_fake_duo = self._duo_forward(
                    gate_real_mid_res[i], gate_fake_mid_res[i], self.out_dim // self.duo_att_ratio, 8, 8)
                gate_mid = torch.nn.CosineSimilarity().forward(gate_real_duo, gate_fake_duo, ).unsqueeze(1)
                gate_mid_res.append(gate_mid)

                for n in range(4): # todo, option, n_downsample
                    # # downsample with interpolate
                    # gate_mid_res[n] = nn.Sequential(
                    #     *[Interpolate(scale_factor=(0.5 ** (4 - n) * 1), mode='nearest')]).forward(gate_mid_res[n])# todo, option, n_downsample

                    # downsample with avgpool net
                    seq = self.pool_of_duo_downsample
                    for i in range(n+1):
                        gate_mid_res[i] = seq.forward(gate_mid_res[i])

                gate_mid = torch.cat(gate_mid_res, 1)
                model = getattr(self, 'conv_af_downsample_gate')
                gate_mid = model.forward(gate_mid)
                # n, c, w, h = gate_mid.size()
                # gate_mid = gate_mid.view(n, c, w * h).permute(0, 2, 1)
                # gate_mid = F.max_pool1d(gate_mid, 6, 1).permute(0, 2, 1).view(n, 1, w, h)
                gate_out = self.out_gated_stream.forward(gate_mid)
            elif self.which_net_mask == 'multiscale5': # downsample, concat, duo, cosine, sim
                for n in range(4): #t odo, option, n_downsample
                    # # downsample with interpolate
                    # seq = nn.Sequential(*[Interpolate(scale_factor=(0.5 ** (4 - n) * 1), mode='nearest')]) # todo, option, n_downsample
                    # gate_fake_mid_res[i], gate_real_mid_res[i] = seq.forward(gate_fake_mid_res[i]), seq.forward(
                    #     gate_real_mid_res[i])

                    # downsample with avgpool net
                    seq = self.pool_of_duo_downsample
                    for i in range(n + 1):
                        gate_fake_mid_res[i], gate_real_mid_res[i] = seq.forward(gate_fake_mid_res[i]), seq.forward(
                                gate_real_mid_res[i])

                gate_fake_mid, gate_real_mid = gate_fake_mid_res[:1+4], gate_real_mid_res[:1+4] # todo, option, 1+n_downsample
                gate_fake_mid.append(gate_fake_mid_res[-1])
                gate_real_mid.append(gate_real_mid_res[-1])

                gate_fake_mid = torch.cat(gate_fake_mid, 1)
                gate_real_mid = torch.cat(gate_real_mid, 1)
                # seq = getattr(self, 'conv_af_downsample_gate')
                # gate_fake_mid = seq.forward(torch.cat(gate_fake_mid, 1))
                # gate_real_mid = seq.forward(torch.cat(gate_real_mid, 1))

                out_dim = self.out_dim * (1+4+1) # todo, option, 1+n_downsample+1
                gate_real_duo, gate_fake_duo = self._duo_forward(
                    gate_real_mid, gate_fake_mid, out_dim // self.duo_att_ratio, 8, 8)
                gate_mid = torch.nn.CosineSimilarity().forward(gate_real_duo, gate_fake_duo, ).unsqueeze(1)
                gate_out = self.out_gated_stream.forward(gate_mid)
            else:
                raise NotImplementedError('mask net name [%s] is not recognized' % self.which_net_mask)

        g_down = self.g_down(input_img)
        g_up = self.g_up(g_down)

        out = g_up * gate_out

        gated_gt = ground_truth * gate_out

        if use_area_constraint:
            gate_sum = gate_mid.sum(3).sum(2)
            return g_up, out, gate_out, gate_sum, gated_gt
        else:
            return g_up, out, gate_out, gated_gt

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


def define_gated_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal',init_gain=0.02, gpu_ids=[], use_gt_mask=False, add_position_signal=True, which_net_mask='basic'):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    netG = GatedGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, use_gt_mask=use_gt_mask, add_position_signal=add_position_signal, which_net_mask=which_net_mask)
    return init_net(netG, init_type, init_gain, gpu_ids)
