import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from util.util import save_image, tensor2im


class SemiAlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')
        assert opt.gt_crop == 1
        # if opt.data_seglabel:
        #     self.dir_seglabel = os.path.join(opt.dataroot, opt.phase+'_label')

        assert not (self.opt.save_data and self.opt.load_data)
        if self.opt.save_data:
            self.dir_AB_save = os.path.join(opt.dataroot, opt.phase+opt.save_dir_suffix)
            assert not os.path.exists(self.dir_AB_save), "do you really want to override the dir %s?" % self.dir_AB_save
            os.mkdir(self.dir_AB_save)
        if self.opt.load_data:
            self.dir_AB_load = os.path.join(opt.dataroot, opt.phase+opt.load_dir_suffix)
            self.AB_paths = sorted(make_dataset(self.dir_AB_load))
    
    def get_A_B(self, index):
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

        # if self.opt.data_seglabel:
        #     image_name = AB_path.split('/')[-1].replace('.jpg', '.png')
        #     Seg = Image.open(os.path.join(self.dir_seglabel, image_name)).resize((self.opt.loadSize, self.opt.loadSize), Image.NEAREST)
        #     Seg = np.array(Seg, dtype="int32")
        #     Seg = Seg[h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        # else:
        #     Seg = None

        return A, B

    def get_default_replace_A_B(self, A, B):
        crop_replace = self.opt.crop_replace

        replace_A = np.ones((A.shape[1], A.shape[2], A.shape[0]))
        replace_B = np.ones((B.shape[1], B.shape[2], B.shape[0]))
        if crop_replace == "white":
            replace_A[:, :, :] = 1.0
            replace_B[:, :, :] = 1.0
        elif crop_replace == "black":
            replace_A[:, :, :] = -1.0
            replace_B[:, :, :] = -1.0
        elif crop_replace == "gray":
            replace_A[:, :, :] = 0.0
            replace_B[:, :, :] = 0.0
        elif crop_replace == "rand":
            replace_A = (np.random.rand(A.shape[1], A.shape[2], A.shape[0]) - 0.5 ) / 0.5
            replace_B = (np.random.rand(B.shape[1], B.shape[2], B.shape[0]) - 0.5 ) / 0.5
        replace_A = transforms.ToTensor()(replace_A).float()
        replace_B = transforms.ToTensor()(replace_B).float()

        return replace_A, replace_B

    def get_replace_A_B(self, cur_idx=[]):
        AB_same_idx = self.opt.replace_ab_same_idx

        index = random.randint(0, len(self.AB_paths) - 1)
        while index in cur_idx:
            index = random.randint(0, len(self.AB_paths) - 1)
        A, B = self.get_A_B(index)

        if not AB_same_idx:
            cur_idx.append(index)
            index = random.randint(0, len(self.AB_paths) - 1)
            while index in cur_idx:
                index = random.randint(0, len(self.AB_paths) - 1)
            _, B = self.get_A_B(index)

        return A, B

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]

        if self.opt.load_data:
            AB = Image.open(AB_path).convert('RGB')

            w, h = AB.size
            w3 = int(w / 3)
            A = AB.crop((0, 0, w3, h))
            B = AB.crop((w3, 0, w3 * 2, h))
            gate = AB.crop((w3 * 2, 0, w, h))

            A = transforms.ToTensor()(A)
            B = transforms.ToTensor()(B)
            gate = transforms.ToTensor()(gate)

            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
            gate = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gate)[0].unsqueeze(0)

        else:
            A, B = self.get_A_B(index)
            if self.opt.gt_crop:
                if self.opt.random_crop == 'random_multiblocks_crop':
                    self.random_crop = self.random_multiblocks_crop
                elif self.opt.random_crop == 'random_oneblock_crop':
                    self.random_crop = self.random_oneblock_crop

                A, B, gate = self.random_crop(A, B, [index])

        # this can be used to generate the test dataset with crop and replacement, and this dataset can be used for all models to compare.
        if self.opt.save_data:
            save_path = os.path.join(self.dir_AB_save, AB_path.split('/')[-1])
            save_AB = torch.zeros((1, A.shape[0], self.opt.fineSize, self.opt.fineSize * 3))
            save_AB[0, :, 0:self.opt.fineSize, 0:self.opt.fineSize] = A
            save_AB[0, :, 0:self.opt.fineSize, self.opt.fineSize:self.opt.fineSize*2] = B
            save_AB[0, 0, 0:self.opt.fineSize, self.opt.fineSize*2:self.opt.fineSize*3] = gate
            save_AB[0, 1, 0:self.opt.fineSize, self.opt.fineSize*2:self.opt.fineSize*3] = gate
            save_AB[0, 2, 0:self.opt.fineSize, self.opt.fineSize*2:self.opt.fineSize*3] = gate
            save_AB_numpy = tensor2im(save_AB)
            save_image(save_AB_numpy, save_path)

        return{'A': A, 'B': B, 'gate':gate, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'SemiAlignedDataset'

    def random_oneblock_crop(self, A, B, cur_idx=[]):
        """Replace arbitrary block in the image"""
        block_h = self.opt.block_size
        block_w = self.opt.block_size
        which_crop = self.opt.which_crop
        crop_replace = self.opt.crop_replace

        h = A.shape[1]
        w = A.shape[2]
        assert h // block_h >= 2 and w // block_w >= 2, "block area should be less than 0.25 of the image"

        gate = np.ones((h, w, 1))
        gate = transforms.ToTensor()(gate)

        if crop_replace == 'image':
            replace_A, replace_B = self.get_replace_A_B(cur_idx)
        else:
            replace_A, replace_B = self.get_default_replace_A_B(A, B)

        h_i = random.randint(0, h-block_h+1)
        w_i = random.randint(0, w-block_w+1)

        if 'A' in which_crop:
            A[:, h_i:h_i+block_h, w_i:w_i+block_w] = replace_A[:, h_i:h_i+block_h, w_i:w_i+block_w]
        if 'B' in which_crop:
            B[:, h_i:h_i+block_h, w_i:w_i+block_w] = replace_B[:, h_i:h_i+block_h, w_i:w_i+block_w]
        gate[:, h_i:h_i+block_h, w_i:w_i+block_w] = 0

        return A, B, gate

    def random_multiblocks_crop(self, A, B, cur_idx=[]):
        """Split the image by size_split * size_split blocks, then choose crop_ratio of them"""
        size_split = self.opt.size_split
        crop_ratio = self.opt.crop_ratio
        which_crop = self.opt.which_crop
        crop_replace = self.opt.crop_replace

        h = A.shape[1]
        w = A.shape[2]

        gate = np.ones((h, w, 1))
        gate = transforms.ToTensor()(gate).float()

        replace_A, replace_B = self.get_default_replace_A_B(A, B)

        num_blocks = size_split * size_split
        assert crop_ratio <= 0.5, "crop_ratio set too high, should be less than 0.5"
        assert h % size_split == 0 and w % size_split == 0, "size_split should be divisible by the sizes of image with h=%d and w=%d" % (h, w)
        num_crops = int(num_blocks * crop_ratio)
        assert num_crops >= 1, "crop_ratio set too low, leading to 0 number of crops"

        block_h = h // size_split
        block_w = w // size_split

        block_indices = np.arange(num_blocks)
        random.shuffle(block_indices)

        for i in range(num_crops):
            h_i = block_indices[i] // size_split * block_h
            w_i = block_indices[i] % size_split * block_w
            
            if crop_replace == 'image':
                replace_A, replace_B = self.get_replace_A_B(cur_idx)

            if 'A' in which_crop:
                A[:, h_i:h_i+block_h, w_i:w_i+block_w] = replace_A[:, h_i:h_i+block_h, w_i:w_i+block_w]
            if 'B' in which_crop:
                B[:, h_i:h_i+block_h, w_i:w_i+block_w] = replace_B[:, h_i:h_i+block_h, w_i:w_i+block_w]
            gate[:, h_i:h_i+block_h, w_i:w_i+block_w] = 0

        return A, B, gate
