import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


class InpaintingDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A, gate = self.crop_center(A)

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            gate = gate.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'gate':gate,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'InpaintingDataset'

    def get_rand_A(self):
        index = random.randint(0, len(self.AB_paths) - 1)
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)

        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        return A

    def crop_center(self, img):
        h = img.shape[1]
        w = img.shape[2]
        h_1 = h // 4
        w_1 = w // 4
        img[:, h_1:h_1 + h // 2, w_1:w_1 + w // 2] = 1 # 1 for all white, 0 for all black

        gate = np.ones((h, w, 1))
        gate = transforms.ToTensor()(gate)
        gate[:, h_1:h_1 + h // 2, w_1:w_1 + w // 2] = 0
        return img, gate

    def random_crop(self, img, ratio=4):
        # Split the image by 4 parts, then choose one
        img = img.clone()
        h = img.shape[1]
        w = img.shape[2]
        gate = np.ones((h, w, 1))
        gate = transforms.ToTensor()(gate)

        i = 0
        while i != ratio:
            h_1 = random.randint(0, ratio - 1)
            w_1 = random.randint(0, ratio - 1)
            h_1 *= h // ratio
            w_1 *= w // ratio
            try:
                img[:, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = 1

                # crop_color = img[:, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio].clone().mean(-1).mean(-1)
                # img[0, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = crop_color[0]
                # img[1, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = crop_color[1]
                # img[2, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = crop_color[2]

                # fake_img = self.get_rand_A()
                # crop_color = fake_img[:, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio].clone().mean(-1).mean(-1)
                # img[0, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = crop_color[0]
                # img[1, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = crop_color[1]
                # img[2, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = crop_color[2]

                gate[:, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = 0
                i += 1
            except Exception:
                print('_aligned_random_crop failed')
                print(h_1)
                print(h // ratio)
                print(img.shape)
                continue
        return img, gate



