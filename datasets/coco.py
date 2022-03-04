import numpy as np
from numpy.lib.utils import deprecate
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio
from . import transforms
import torchvision

def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list

def load_cls_label_list(name_list_dir):
    
    return np.load(os.path.join(name_list_dir,'cls_labels_onehot.npy'), allow_pickle=True).item()

def robust_read_image(image_name):
    image = np.asarray(imageio.imread(image_name))
    if len(image.shape)<3:
        image = np.stack((image, image, image), axis=-1)
    return image


class CocoDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        name_list_dir=None,
        split='train',
        stage='train',
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.label_dir = os.path.join(root_dir, 'SegmentationClass')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

        if "train" in split:
            self.label_dir = os.path.join(self.label_dir, "train")
            self.img_dir = os.path.join(self.img_dir, "train")
        elif "val" in split:
            self.label_dir = os.path.join(self.label_dir, "val")
            self.img_dir = os.path.join(self.img_dir, "val")

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name+'.jpg')
        #image = np.asarray(imageio.imread(img_name))
        image = robust_read_image(img_name)

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "test":
            label = image[:,:,0]

        return _img_name, image, label


class CocoClsDataset(CocoDataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image):
        img_box = None
        if self.aug:
            '''
            if self.resize_range: 
                image, label = transforms.random_resize(
                    image, label, size_range=self.resize_range)
            '''
            if self.rescale_range:
                image = transforms.random_scaling(
                    image,
                    scale_range=self.rescale_range)
            
            if self.img_fliplr:
                image = transforms.random_fliplr(image)
            #image = self.color_jittor(image)
            if self.crop_size:
                image, img_box = transforms.random_crop(
                    image,
                    crop_size=self.crop_size,
                    mean_rgb=[0,0,0],#[123.675, 116.28, 103.53], 
                    ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
        image = transforms.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, img_box

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        #label_onehot = F.one_hot(label, num_classes)
        
        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):

        img_name, image, _ = super().__getitem__(idx)

        image, img_box = self.__transforms(image=image)

        cls_label = self.label_list[img_name]

        if self.aug:
            return img_name, image, cls_label, img_box
        else:
            return img_name, image, cls_label


class CocoSegDataset(CocoDataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            '''
            if self.resize_range: 
                image, label = transforms.random_resize(
                    image, label, size_range=self.resize_range)
            
            if self.rescale_range:
                image, label = transforms.random_scaling(
                    image,
                    label,
                    scale_range=self.rescale_range)
            '''
            if self.img_fliplr:
                image, label = transforms.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label = transforms.random_crop(
                    image,
                    label,
                    crop_size=self.crop_size,
                    mean_rgb=[123.675, 116.28, 103.53], 
                    ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
        image = transforms.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)

        image, label = self.__transforms(image=image, label=label)

        cls_label = self.label_list[img_name]

        return img_name, image, label, cls_label
