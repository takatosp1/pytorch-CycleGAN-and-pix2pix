import os.path
import random
import torchvision.transforms as transforms
import torch
import random
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')
        self.clip_size = opt.clip_size

    def _get_rand_A(self):
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

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
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

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)
        return A


    def __getitem__(self, index_):
        index = int(index_ / self.clip_size)
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

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        if self.opt.aligned_random_crop:
            list_A = []
            list_gate = []
            for num_stream in range(self.opt.num_stream):
                tmp, gate_tmp = self._aligned_random_crop(A.clone())
                list_A.append(tmp)
                list_gate.append(gate_tmp)
            A = list_A
            gate = list_gate

        return {'A': A, 'B': B, 'gate': gate,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths) * self.clip_size

    def name(self):
        return 'AlignedDataset'

    def _jitter(img):
        # dimension: C x H x W
        pass

    def _aligned_random_crop(self, img):
        fake_A = self._get_rand_A()
        h = img.shape[1]
        w = img.shape[2]
        h_1 = random.randint(1, h // 2 - 1)
        w_1 = random.randint(1, w // 2 - 1)
        gate = np.zeros((h, w, 1))

        try:
            img[:, :h_1, :] = fake_A[:, :h_1, :]
            img[:, :, :w_1] = fake_A[:, :, :w_1]
            img[:, h_1 + h // 2:, :] = fake_A[:, h_1 + h // 2:, :]
            img[:, :, w_1 + w // 2:] = fake_A[:, :, w_1 + w // 2:]
            gate[:, :h_1, :] = 0
            gate[:, :, :w_1] = 0
            gate[:, h_1 + h // 2:, :] = 0
            gate[:, :, w_1 + w // 2:] = 0

        except Exception:
            print('_aligned_random_crop failed')
            print(h_1)
            print(h // 2)
            print(img.shape)
        gate = transforms.ToTensor()(gate)
        return img, gate
