"""
Code from Reinforcement Learning with Augmented Data

Source link: https://github.com/pokaxpoka/rad_procgen

Modify Rand_Crop: first resize obs to (75, 75) then perform random crop,
                  otherwise cropping is ill-defined
"""

import numpy as np
import tensorflow as tf

from skimage.util.shape import view_as_windows

class Cutout_Color(object):
    def __init__(self, 
                 batch_size, 
                 box_min=7, 
                 box_max=22, 
                 pivot_h=12, 
                 pivot_w=24, 
                 obs_dtype='uint8', 
                 *_args, 
                 **_kwargs):
        
        self.box_min = box_min
        self.box_max = box_max
        self.pivot_h = pivot_h
        self.pivot_w = pivot_w
        self.batch_size = batch_size
        self.w1 = np.random.randint(self.box_min, self.box_max, batch_size)
        self.h1 = np.random.randint(self.box_min, self.box_max, batch_size)
        self.rand_box = np.random.randint(0, 255, size=(batch_size, 1, 1, 3), dtype=obs_dtype)
        self.obs_dtype = obs_dtype
        
    def do_augmentation(self, imgs):
        n, h, w, c = imgs.shape
        pivot_h = 12
        pivot_w = 24

        cutouts = np.empty((n, h, w, c), dtype=imgs.dtype)
        for i, (img, w11, h11) in enumerate(zip(imgs, self.w1, self.h1)):
            cut_img = img.copy()
            cut_img[self.pivot_h+h11:self.pivot_h+h11+h11, self.pivot_w+w11:self.pivot_w+w11+w11, :] \
            = np.tile(self.rand_box[i], cut_img[self.pivot_h+h11:self.pivot_h+h11+h11, 
                                                self.pivot_w+w11:self.pivot_w+w11+w11, :].shape[:-1] +(1,))
            cutouts[i] = cut_img
        return cutouts
        
    def change_randomization_params(self, index_):
        self.w1[index_] = np.random.randint(self.box_min, self.box_max)
        self.h1[index_] = np.random.randint(self.box_min, self.box_max)
        self.rand_box[index_] = np.random.randint(0, 255, size=(1, 1, 1, 3), dtype=self.obs_dtype)

    def change_randomization_params_all(self):
        self.w1 = np.random.randint(self.box_min, self.box_max, self.batch_size)
        self.h1 = np.random.randint(self.box_min, self.box_max, self.batch_size)
        self.rand_box = np.random.randint(0, 255, size=(self.batch_size, 1, 1, 3), dtype=self.obs_dtype)
        
    def print_parms(self):
        print(self.w1)
        print(self.h1)

class Rand_Crop(object):
    def __init__(self,  
                 batch_size,
                 sess,
                 *_args, 
                 **_kwargs):
        
        self.batch_size = batch_size
        self.crop_size = 64
        self.crop_max = 75 - self.crop_size
        self.sess = sess
        self.imgs_ph = tf.placeholder(shape=(batch_size, 64, 64, 3), dtype=np.uint8)
        self.resized_imgs_ph = tf.image.resize_images(self.imgs_ph, size=(75, 75))
        self.w1 = np.random.randint(0, self.crop_max, self.batch_size)
        self.h1 = np.random.randint(0, self.crop_max, self.batch_size)

    def do_augmentation(self, imgs):
        # batch size
        n = imgs.shape[0]
        img_size = imgs.shape[1]

        # resize to (75, 75)
        imgs = self.sess.run(self.resized_imgs_ph, {self.imgs_ph: imgs})
        imgs = imgs.astype('uint8')
        
        # transpose
        imgs = np.transpose(imgs, (0, 2, 1, 3))
        
        # creates all sliding windows combinations of size (output_size)
        windows = view_as_windows(
            imgs, (1, self.crop_size, self.crop_size, 1))[..., 0,:,:, 0]
        # selects a random window for each batch element
        cropped_imgs = windows[np.arange(n), self.w1, self.h1]
        cropped_imgs = np.swapaxes(cropped_imgs,1,3)
        return cropped_imgs
    
    def change_randomization_params(self, index_):
        self.w1[index_] = np.random.randint(0, self.crop_max)
        self.h1[index_] = np.random.randint(0, self.crop_max)

    def change_randomization_params_all(self):
        self.w1 = np.random.randint(0, self.crop_max, self.batch_size)
        self.h1 = np.random.randint(0, self.crop_max, self.batch_size)
    
    def print_parms(self):
        print(self.w1)
        print(self.h1)
