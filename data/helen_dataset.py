##############################################################################
# The code is not yet fully updated and refined
# TODO(yi 2020): investigate it
##############################################################################
import os.path
import random
import torchvision.transforms as transforms
import torch
import random
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

colormap = np.asarray([
    [0, 0, 0],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [70, 130, 180],
    [107, 142, 35],
    [0, 0, 142],
    [152, 251, 152],
    [220, 20, 60],
    ])


class HelenDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.crop_label_replace = opt.crop_label_replace
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert (opt.resize_or_crop == 'resize_and_crop')

        self.dir_A = os.path.join(opt.dataroot, opt.phase+'A')
        self.dir_seglabel = os.path.join(opt.dataroot, opt.phase+'Label')
        self.A_paths = sorted(make_dataset(self.dir_A))

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        image_name = A_path.split('/')[-1]
        A = np.array(Image.open(A_path).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC))
        seg = np.array(Image.open(os.path.join(self.dir_seglabel, image_name)).resize((self.opt.loadSize, self.opt.loadSize), Image.NEAREST))

        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        # A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        A = A[h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize, :]
        seg = seg[h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        if self.opt.which_crop == 'B':
            seg, gate = self.random_crop_change_label(seg)
            #  _, B, gate = self.random_replace_nosemouth(seg)
        elif self.opt.which_crop == 'A':
            A, gate = self.random_crop_image_change_color(A, seg)
            # A, gate = self.random_crop_paste_image(A, seg)

        gate = transforms.ToTensor()(gate)
        A = transforms.ToTensor()(A)
        B = colormap[seg]/255.0

        B = transforms.ToTensor()(B).type(torch.FloatTensor)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B) # why????

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
            # gate = gate.index_select(1, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)
        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        AB_path = self.AB_paths[index]
        return {'A': A, 'B': B, 'gate': gate,
                'A_paths': AB_path, 'B_paths': AB_path}
    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'HelenDataset'

    def get_rand_A(self):
        index = random.randint(0, len(self.AB_paths) - 1)
        A_path = self.A_paths[index]
        A = np.array(Image.open(A_path).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC))
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        A = A[h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize, :]
        return A

    def random_crop_change_label(self, seg, ratio=4):
        h, w = np.shape(seg)
        gate = np.ones((h, w, 1))

        i = 0
        ratio=2
        while i != 1:
            h_1 = random.randint(0, ratio - 1)
            w_1 = random.randint(0, ratio - 1)
            h_1 *= h // ratio
            w_1 *= w // ratio
            # h_1 = random.randint(0, 4) * h // 8
            # w_1 = random.randint(0, 4) * h // 8

            try:
                seg_crop = seg[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio].flatten()
                seg_crop_lab = sorted(np.unique(seg_crop))
                if len(seg_crop_lab) == 1 and (seg_crop_lab[0] == 0 or seg_crop_lab[0] == 1) or \
                        len(seg_crop_lab) == 2 and (seg_crop_lab[0] == 0 or seg_crop_lab[0] == 1):
                    continue
                if self.crop_label_replace=='random':
                    label = random.choice(
                        list(set(np.arange(11)) - set(seg_crop_lab) | {0, 1}))  # 11 label classes for helen
                elif self.crop_label_replace=='face':
                    label = 1
                elif self.crop_label_replace=='facial':
                    choice_list = list({2,3,4,5,6,7,8,9,10} - set(seg_crop_lab))
                    if len(choice_list) == 0:
                        continue
                    label = random.choice(choice_list)
                else:
                    label = 0
                # ind = list(set(np.where(seg_crop != 0)[0]) & set(np.where(seg_crop != 1)[0]))
                # seg_crop[ind] = label
                # seg[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = seg_crop.reshape((h // ratio, w // ratio))
                seg[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio] = label
                gate[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio, :] = 0  # 1 means true
                i += 1
            except Exception:
                print('_aligned_random_crop failed')
                print(h_1, w_1)
                print(h // ratio, w//ratio)
                print(np.shape(seg))

        return seg, gate

    def random_crop_image_change_color(self, A, seg, ratio=4):
        h, w = np.shape(seg)
        gate = np.ones((h, w, 1))

        mask = np.zeros(np.shape(seg))
        ind = np.where(seg == 1)
        mask[ind] = 1 # face
        A_ = A * np.expand_dims(mask, axis=-1)
        avg_face_color = [A_[:,:,0].sum()/len(ind[0]), A_[:,:,1].sum()/len(ind[0]), A_[:,:,2].sum()/len(ind[0])]

        i = 0
        ratio = 2
        while i != 1:
            h_1 = random.randint(0, ratio - 1)
            w_1 = random.randint(0, ratio - 1)
            h_1 *= h // ratio
            w_1 *= w // ratio

            try:
                seg_crop = seg[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio].flatten()
                seg_crop_lab = sorted(np.unique(seg_crop))
                if len(seg_crop_lab) == 1 and (seg_crop_lab[0] == 0 or seg_crop_lab[0] == 1) or \
                        len(seg_crop_lab) == 2 and (seg_crop_lab[0] == 0 or seg_crop_lab[0] == 1):
                    continue
                A[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio, :] = avg_face_color
                gate[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio, :] = 0  # 1 means true
                i += 1
            except Exception:
                print('_aligned_random_crop failed')
                print(h_1, w_1)
                print(h // ratio, w//ratio)
                print(np.shape(seg))

        return A, gate

    def random_crop_paste_image(self, img, seg, ratio=4):
        h = img.shape[0]
        w = img.shape[1]

        gate = np.ones((h, w, 1))

        ratio = 2
        i=0
        while i != 1:
            rand_A = self.get_rand_A()
            h_1 = random.randint(0, ratio - 1)
            w_1 = random.randint(0, ratio - 1)
            h_1 *= h // ratio
            w_1 *= w // ratio

            try:
                # seg_crop = seg[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio].flatten()
                # seg_crop_lab = sorted(np.unique(seg_crop))
                # if len(seg_crop_lab) == 1 and (seg_crop_lab[0] == 0 or seg_crop_lab[0] == 1) or \
                #         len(seg_crop_lab) == 2 and (seg_crop_lab[0] == 0 or seg_crop_lab[0] == 1):
                #     continue
                img[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio, :] = \
                        rand_A[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio, :]
                gate[h_1:h_1 + h // ratio, w_1:w_1 + w // ratio, :] = 0 # 1 means true
                i += 1
            except Exception:
                print('_aligned_random_crop failed')
                print(h_1)
                print(h // ratio)
                print(img.shape)
        return img, gate

    def random_replace_nosemouth(self, seg):
        h, w = np.shape(seg)
        gate = np.ones((h, w, 1))
        gate = transforms.ToTensor()(gate)
        if random.random()<0.5:
            for i in [6, 7, 8, 9]:  #
                ind = np.where(seg==i)
                if len(ind[0]) == 0:
                    continue
                max_h = np.max(ind[0]) + 1
                min_h = np.min(ind[0])
                max_w = np.max(ind[1]) + 1
                min_w = np.min(ind[1])
                seg[min_h: max_h, min_w: max_w] = 1 # face label
                gate[:, min_h: max_h, min_w: max_w] = 0 # 1 means true
                # seg[ind] = 1
                # gate[ind = 1
        B = colormap[seg] / 255.0
        B = transforms.ToTensor()(B).type(torch.FloatTensor)
        return seg, B, gate