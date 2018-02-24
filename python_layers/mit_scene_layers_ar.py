import caffe
import time
import numpy as np
from PIL import Image
import scipy.io
import scipy.misc
import cv2
import math

import random

class MITSceneDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.img_dir = params['img_dir']
        self.img_num = params['img_num']
        self.ann_dir = params['ann_dir']
        self.split = params['split']
        self.mean = np.array((109.5388, 118.6897, 124.6901), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # crop size
        self.crop_size = params['crop_size']

        # scales
        self.scale_choices = (0.5, 0.75, 1.0, 1.25, 1.75)


        # 4 tops: data and label
        if len(top) != 8:
            raise Exception("Need to define 8 tops: data and label...etc")


        # load indices for images and labels
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            np.random.seed(self.seed)
            self.idx_list = np.random.permutation(self.img_num)
            #self.idx = random.randint(1, self.img_num)
        else:
            self.idx_list = np.arange(self.img_num)

        # prepare feature for encoding
        self.enc_dir = params['enc_dir']
        self.enc_mat = params['enc_mat']

        # prepare prior for encoding
        self.prior_dir = params['prior_dir']


        # prior constants
        self.normalize_k = 1.0/5.0
        self.normalize_k_global = 1.0/5.0/384.0/384.0

    def reshape(self, bottom, top):
        self.flip = False
        if len(bottom) != 0:
            self.crop_size = bottom[0].data.flat[0]
            self.idx = bottom[0].data.flat[1]
            if self.crop_size < 1:
                self.crop_size = 321
            self.flip = bottom[0].data.flat[2] > 0
        
        img_idx = self.idx_list[self.idx] + 1
        

        # random select one scale
        self.scale = random.choice(self.scale_choices)
        # load image + label image pair


        #t = time.time()
        self.data = self.load_image(img_idx)
        self.label = self.load_label(img_idx)
        #print 'loading image/label {} sec'.format(time.time()- t) 

        # load feature for encoding
        #print 'loading siamese feature'
        #t = time.time()
        self.conv5_3_cxt, self.fc6_cxt, self.fc7_cxt = self.load_feat_enc(img_idx)
        #print '{} sec'.format(time.time()- t)

        # load prior
        # print 'loading prior'
        # t = time.time()
        #self.prior_spatial_size = int(math.ceil(float(self.crop_size)/8))
        self.prior_spatial, self.prior_global = self.load_prior(img_idx)
        #print 'loading prior {} sec'.format(time.time()- t)
        #print 'finish loading prior'

        #random flip image
        if self.flip or self.split == 'train' and random.randint(0,1):
            print 'flip'
            self.data = self.data[:,:,::-1]
            self.label = self.label[:,:,::-1]
            self.prior_spatial = self.prior_spatial[:,:,::-1]

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
        top[2].reshape(*self.conv5_3_cxt.shape)
        top[3].reshape(*self.fc6_cxt.shape)
        top[4].reshape(*self.fc7_cxt.shape)
        top[5].reshape(*self.prior_spatial.shape)
        top[6].reshape(*self.prior_global.shape)
        
        #data_dim
        top[7].reshape(1,1,1,2)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        top[2].data[...] = self.conv5_3_cxt
        top[3].data[...] = self.fc6_cxt
        top[4].data[...] = self.fc7_cxt
        top[5].data[...] = self.prior_spatial
        top[6].data[...] = self.prior_global        

        top[7].data[...] = self.crop_size, self.crop_size

        # pick next input
        self.idx += 1
        if self.idx == self.img_num:
            self.idx = 0
            if self.random:
                # re-shuffle
                self.idx_list = np.random.permutation(self.img_num)

        #print 'data layer forward()'

    def backward(self, top, propagate_down, bottom):
        #print 'data layer backward()'
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/ADE_{}_'.format(self.img_dir, self.split)+'{0:0>8}.jpg'.format(idx))
        in_ = np.array(im, dtype=np.float32)

        if in_.ndim == 2:
            #BW image
            in_ = in_[:,:,np.newaxis]
            in_ = np.repeat(in_,3,axis=2)

        # RGB -> BGR
        in_ = in_[:,:,::-1]

        s = self.crop_size

        if self.split is 'train':

            #rescale
            self.new_shape = tuple(int(x*self.scale) for x in in_.shape[0:2])
            in_ = cv2.resize(in_, self.new_shape[::-1], interpolation=cv2.INTER_LINEAR)

            #check if we need to pad image
            pad_h = max(s-self.new_shape[0], 0)
            pad_w = max(s-self.new_shape[1], 0)

            # padding
            in_ = cv2.copyMakeBorder(in_,0,pad_h,0,pad_w, cv2.BORDER_CONSTANT,
                         value = (109.5388, 118.6897, 124.6901))

            # radom select upperleft crop point
            x_top = random.randint(0,in_.shape[0]-s)
            y_top = random.randint(0,in_.shape[1]-s)

            self.crop_x = random.randint(0,x_top)
            self.crop_y = random.randint(0,y_top)
            x = self.crop_x
            y = self.crop_y

            in_ = in_[x:x+s,y:y+s,:]

        else:
            h, w = in_.shape[0:2]
            if h > w:
                self.new_shape = (s, int(w*s/h)/32*32+1)
            else:
                self.new_shape = (int(h*s/w)/32*32+1,s)
            in_ = cv2.resize(in_, self.new_shape[::-1], interpolation=cv2.INTER_LINEAR)

            #in_ = cv2.copyMakeBorder(in_,0,s-self.new_shape[0],0,s-self.new_shape[1], cv2.BORDER_CONSTANT,
            #           value = (109.5388, 118.6897, 124.6901))

            in_ = np.array(in_, dtype=np.float32)

        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        
        """
        im = Image.open('{}/ADE_{}_'.format(self.ann_dir,self.split)+'{0:0>8}.png'.format(idx))
        label = np.array(im, dtype=np.uint8)

        s = self.crop_size

        if self.split is 'train':
            #new_shape =  tuple(int(x*self.scale) for x in label.shape[0:2])
            label = cv2.resize(label, self.new_shape[::-1], interpolation=cv2.INTER_NEAREST)

            #check if we need to pad image
            pad_h = max(s-self.new_shape[0], 0)
            pad_w = max(s-self.new_shape[1], 0)

            #padding
            label = cv2.copyMakeBorder(label,0,pad_h,0,pad_w, cv2.BORDER_CONSTANT, value = 255)


            # radom select upperleft crop point
            x = self.crop_x
            y = self.crop_y


            label = label[x:x+s,y:y+s]


        label = label[np.newaxis, ...]

        #transform 0 to 255
        label[label==0] = 255
        label[label!=255] -= 1

        return label

    def load_feat_enc(self, idx):
        filename = 'ADE_{0:0>8}'.format(idx)

        fc7_cxt = scipy.io.loadmat('{}/{}/{}.mat'.format(self.enc_dir, self.enc_mat.format('fc7'), filename))['fc7']
        return np.zeros((1,)), np.zeros((1,)), fc7_cxt

    def load_prior(self, idx):

        prefix = 'ADE_{}_'.format(self.split)
        filename = prefix + '{0:0>8}'.format(idx)

        #print 'loading {}'.format(filename)
        mat_data = scipy.io.loadmat('{}/{}.mat'.format(self.prior_dir, filename))

        # Spatial Prior
        #t = time.time()
        prior_spatial = mat_data['prior_spatial']

        #rescale and crop spatial prior
        if self.split is 'train':

            p_s_size = prior_spatial.shape[1] 

            ratio_h = float(p_s_size) / self.new_shape[0]
            ratio_w = float(p_s_size) / self.new_shape[1]

            x = int(self.crop_x * ratio_h)
            y = int(self.crop_y * ratio_w)
            crop_h = int(self.crop_size * ratio_h)
            crop_w = int(self.crop_size * ratio_w)

            #print self.new_shape, self.crop_x, self.crop_y, x, y, crop_h, crop_w

            crop_prior = np.zeros((prior_spatial.shape[0], crop_h, crop_w))
            
            crop_h = min(x+crop_h, p_s_size) - x
            crop_w = min(y+crop_w, p_s_size) - y

            crop_prior[:, 0:crop_h, 0:crop_w] = prior_spatial[:,  x:x+crop_h, y:y+crop_w]

            prior_spatial = cv2.resize(crop_prior.transpose((1,2,0)),
                                       (p_s_size, p_s_size), 
                                       interpolation=cv2.INTER_LINEAR).transpose((2,0,1))

        else: #resize to according size
            prior_spatial = cv2.resize(prior_spatial.transpose((1,2,0)),
                                       (int(math.ceil(float(self.new_shape[1])/8)), int(math.ceil(float(self.new_shape[0])/8))),
                                       interpolation=cv2.INTER_LINEAR).transpose((2,0,1))


        prior_spatial = prior_spatial[np.newaxis, ...]
        prior_spatial = np.multiply(prior_spatial, self.normalize_k)

        # Global Prior
        #t = time.time()
        prior_global = mat_data['prior_global'] * self.normalize_k_global
        prior_global = prior_global[np.newaxis, ...]

        return prior_spatial, prior_global
