import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from random import randrange
import torch
from glob import glob
import os.path as osp
import json

class PairDataset(Dataset):

    def __init__(self, first_dir, second_dir, third_dir, num_samples):
        """
        Arguments:
            first_dir, second_dir: strings, paths to folders with images.
            num_samples: an integer.
            image_size: a tuple of integers (height, width).
        """

        # Dehazing
        self.first_dir = first_dir + 'train/'

        self.haze_dir = []
        for root, dirs, files in os.walk(self.first_dir):
            for fname in files:
                self.haze_dir.append(os.path.join(root, fname))

        # GT
        self.second_dir = second_dir + 'train/'

        self.gt_dir = []
        for root, dirs, files in os.walk(self.second_dir):
            for fname in files:
                for i in range(3):
                    self.gt_dir.append(os.path.join(root, fname))

        # Mask
        self.third_dir = third_dir

        with open('./BiseNet/cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse gt directory
        self.mask_dir = []
        gtnames = []
        # gtpth = osp.join('D:\Dataset\DBF_trainval_foggy', 'gtFine', mode)
        gtpth = osp.join('D:\Dataset\DBF_trainval_foggy', 'gtFine', 'train')
        folders = os.listdir(gtpth)

        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]

            new_lbnames = []
            for name in lbnames:
                for i in range(3):
                    new_lbnames.append(name)

            names = [el.replace('_gtFine_labelIds.png', '') for el in new_lbnames]
            lbpths = [osp.join(fdpth, el) for el in new_lbnames]
            gtnames.extend(names)
            self.mask_dir.extend(lbpths)

        self.num_samples = num_samples

        self.transform = transforms.Compose([
            transforms.Resize((512, 1024), 3),
            transforms.RandomCrop(360),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((512, 1024), 0),
            transforms.RandomCrop(360),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ])

        self.augmentation = transforms.RandomRotation(45, resample=Image.BICUBIC)

        self.colorjitter = transforms.ColorJitter(brightness=0.2)

        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.totensor_gt = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __len__(self):
        return self.num_samples

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label

    def __getitem__(self, _):
        """
        Get a random pair of image crops.
        It returns a tuple of float tensors with shape [3, height, width].
        They represent RGB images with pixel values in [0, 1] range.
        """

        # Cityscape
        i = np.random.randint(0, len(self.haze_dir))

        # Haze ---------------------------------------------------------------------------------------------------------
        haze = Image.open(self.haze_dir[i])
        gt = Image.open(self.gt_dir[i])
        mask = Image.open(self.mask_dir[i])

        # try:
        #     haze_gt = Image.open(os.path.join(self.second_dir, name2) + '.jpg').convert('RGB')
        # except:
        #     haze_gt = Image.open(os.path.join(self.second_dir, name2) + '.png').convert('RGB')

        seed = np.random.randint(654235)

        random.seed(seed)
        haze = self.transform(haze)

        random.seed(seed)
        gt = self.transform(gt)

        random.seed(seed)
        mask = self.transform_mask(mask)

        haze = self.totensor_gt(haze)
        gt = self.totensor_gt(gt)

        mask = np.array(mask).astype(np.int64)[np.newaxis, :]
        mask = self.convert_labels(mask)

        return haze, gt, mask

