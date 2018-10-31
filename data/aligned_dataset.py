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
        if opt.data_seglabel:
            self.dir_seglabel = os.path.join(opt.dataroot, opt.phase+'_label')

    def get_rand_A(self, which='A'):
        index = random.randint(0, len(self.AB_paths) - 1)
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        if which=='A':
            A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        elif which =='B':
            A = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
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

        if self.opt.data_seglabel:
            image_name = AB_path.split('/')[-1].replace('.jpg', '.png')
            Seg = Image.open(os.path.join(self.dir_seglabel, image_name)).resize((self.opt.loadSize, self.opt.loadSize), Image.NEAREST)
            Seg = np.array(Seg, dtype="int32")
            Seg = Seg[h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        else:
            Seg = None

        return A, Seg

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

        if self.opt.data_seglabel:
            image_name = AB_path.split('/')[-1].replace('.jpg', '.png')
            Seg = Image.open(os.path.join(self.dir_seglabel, image_name)).resize((self.opt.loadSize, self.opt.loadSize), Image.NEAREST)
            Seg = np.array(Seg, dtype="int32")
            Seg = Seg[h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        else:
            Seg = None

        if self.opt.gt_crop:
            if self.opt.random_crop == 'random_multiblocks_crop':
                self.random_crop = self.random_multiblocks_crop
            elif self.opt.random_crop == 'random_oneblock_crop':
                self.random_crop = self.random_oneblock_crop

            if self.opt.which_crop =='A':
                A, gate = self.random_crop(A.clone(), 'A', Seg)
            elif self.opt.which_crop =='B':
                B, gate = self.random_crop(B.clone(), 'B', Seg)

            return {'A': A, 'B': B, 'gate': gate,
                    'A_paths': AB_path, 'B_paths': AB_path}
        else:
            return {'A': A, 'B': B,
                    'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'

    def random_multiblocks_crop(self, img, which='A', seg=None, ratio=4):
        # Split the image by 4 parts, then choose one
        img = img.clone()
        h = img.shape[1]
        w = img.shape[2]

        gate = np.ones((h, w, 1))
        gate = transforms.ToTensor()(gate)

        i=0
        while i != ratio:
            fake_A, seg_A = self.get_rand_A(which)
            h_1 = random.randint(0, ratio - 1)
            w_1 = random.randint(0, ratio - 1)
            h_1 *= h // ratio
            w_1 *= w // ratio
            try:
                if seg is not None:
                    seg_crop_lab = np.unique(seg[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio].squeeze())
                    seg_A_crop_lab = np.unique(seg_A[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio].squeeze())
                    if len(seg_crop_lab) == 1 and len(seg_A_crop_lab) == 1:
                        if seg_crop_lab[0] == seg_A_crop_lab[0]:
                            # which means the cropped region label is exactly the same as the random one
                            continue
                img[:, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = \
                        fake_A[:, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio]
                gate[:, h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = 0 # 1 means true
                i += 1
            except Exception:
                print('_aligned_random_crop failed')
                print(h_1)
                print(h // ratio)
                print(img.shape)
        return img, gate

    def random_oneblock_crop(self, img, which='A',  oneD_ratio=2):
        # Split the image by 4 parts, then choose one
        fake_A = self.get_rand_A(which)
        h = img.shape[1]
        w = img.shape[2]
        h_1 = random.randint(0, oneD_ratio-1)
        w_1 = random.randint(0, oneD_ratio-1)
        h_1 *= h // oneD_ratio
        w_1 *= w // oneD_ratio
        gate = np.ones((h, w, 1))
        gate = transforms.ToTensor()(gate)
        #gate = np.zeros((h,w,1))
        #gate = transforms.ToTensor()(gate)

        try:
            img[:, h_1:h_1 + h // oneD_ratio, w_1:w_1 + w // oneD_ratio] = \
                    fake_A[:, h_1:h_1 + h // oneD_ratio, w_1:w_1 + w // oneD_ratio]
            gate[:, h_1:h_1 + h // oneD_ratio, w_1:w_1 + w // oneD_ratio] = 0 #1 menas true
            #fake_A[:, h_1:h_1 + h // oneD_ratio, w_1:w_1 + w // oneD_ratio] = img[:, h_1:h_1 + h // oneD_ratio, w_1:w_1 + w // oneD_ratio]
            #gate[:, h_1:h_1 + h // oneD_ratio, w_1:w_1 + w // oneD_ratio] = 1
        except Exception:
            print('_aligned_random_crop failed')
            print(h_1)
            print(h // oneD_ratio)
            print(img.shape)
        return img, gate
        #return fake_A, gate
