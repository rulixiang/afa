import random
import numpy as np
from PIL import Image
#from scipy import misc
import mmcv
import imageio

def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    imgarr = np.asarray(img)
    proc_img = np.empty_like(imgarr, np.float32)

    proc_img[..., 0] = (imgarr[..., 0] - mean[0]) / std[0]
    proc_img[..., 1] = (imgarr[..., 1] - mean[1]) / std[1]
    proc_img[..., 2] = (imgarr[..., 2] - mean[2]) / std[2]
    return proc_img

def random_scaling(image, label=None, scale_range=None):

    min_ratio, max_ratio = scale_range
    assert min_ratio <= max_ratio

    ratio = random.uniform(min_ratio, max_ratio)

    return _img_rescaling(image, label, scale=ratio)

def _img_rescaling(image, label=None, scale=None):
    
    #scale = random.uniform(scales)
    h, w, _ = image.shape
    
    new_scale = [int(scale * w), int(scale * h)]

    new_image = Image.fromarray(image.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_image = np.asarray(new_image).astype(np.float32)

    if label is None:
        return new_image

    new_label = Image.fromarray(label).resize(new_scale, resample=Image.NEAREST)
    new_label = np.asarray(new_label)
    
    return new_image, new_label

def img_resize_short(image, min_size=512):
    h, w, _ = image.shape
    if min(h, w) >= min_size:
        return image

    scale = float(min_size) / min(h, w)
    new_scale = [int(scale * w), int(scale * h)]

    new_image = Image.fromarray(image.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_image = np.asarray(new_image).astype(np.float32)

    return new_image

def random_resize(image, label=None, size_range=None):
    _new_size = random.randint(size_range[0], size_range[1])
  
    h, w, _ = image.shape
    scale = _new_size / float(max(h, w))
    new_scale = [int(scale * w), int(scale * h)]
    
    return _img_rescaling(image, label, scale=new_scale)

def random_fliplr(image, label=None):
    p = random.random()

    if label is None:
        if p > 0.5:
            image  = np.fliplr(image)
        return image
    else:
        if p > 0.5:
    
            image = np.fliplr(image)
            label = np.fliplr(label)

        return image, label

def random_flipud(image, label=None):

    p = random.random()

    if label is None:
        if p > 0.5:
            image  = np.flipud(image)
        return image
    else:
        if p > 0.5:
    
            image = np.flipud(image)
            label = np.flipud(label)

        return image, label

def random_rot(image, label):

    k = random.randrange(3) + 1

    image = np.rot90(image, k).copy()

    if label is None:
        return image

    label = np.rot90(label, k).copy()

    return image, label

def random_crop(image, label=None, crop_size=None, mean_rgb=[0,0,0], ignore_index=255):

    h, w, _ = image.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_image = np.zeros((H,W,3), dtype=np.float32)
    
    pad_image[:,:,0] = mean_rgb[0]
    pad_image[:,:,1] = mean_rgb[1]
    pad_image[:,:,2] = mean_rgb[2]
    
    H_pad = int(np.random.randint(H-h+1))
    W_pad = int(np.random.randint(W-w+1))
    
    pad_image[H_pad:(H_pad+h), W_pad:(W_pad+w), :] = image
    
    def get_random_cropbox(_label, cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            if _label is None:
                return H_start, H_end, W_start, W_end, 

            temp_label = _label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]

            if len(cnt>1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end, 

    H_start, H_end, W_start, W_end = get_random_cropbox(label)

    image = pad_image[H_start:H_end, W_start:W_end,:]

    img_H_start = max(H_pad-H_start, 0)
    img_W_start = max(W_pad-W_start, 0)
    img_H_end = min(H_end, H_pad+h)
    img_W_end = min(W_end, W_pad+w)
    img_box = np.asarray([img_H_start, img_H_end, img_W_start, img_W_end], dtype=np.int16)

    if label is None:

        return image, img_box

    pad_label = np.ones((H,W), dtype=np.float32) * ignore_index
    pad_label[H_pad:(H_pad+h), W_pad:(W_pad+w)] = label
    label = pad_label[H_start:H_end, W_start:W_end]

    return image, label, img_box

class PhotoMetricDistortion(object):
    """ from mmseg """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if np.random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if np.random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      np.random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, img):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        #img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        #results['img'] = img
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str