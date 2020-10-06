# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import math
import os
import boto
from numpy import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys
#import resnet_3D
from filechunkio import FileChunkIO
from glob import glob
from scipy.interpolate import spline
import random
from sklearn import preprocessing
from matplotlib import pyplot
from keras import optimizers
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import backend as K
from keras.callbacks import *
from keras.utils import np_utils
import copy
#from Test_Merge_Model_Mutl import DPSEN_Contruct
from Merge_mode import DPSEN_Contruct
import xlrd
import time
import xlwt
from collections import defaultdict
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model
import threading
import warnings
from keras.regularizers import l2
import argparse
from keras.callbacks import Callback, LearningRateScheduler
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import ReduceLROnPlateau, CSVLogger, TensorBoard, EarlyStopping

# GPU 配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 每个GPU 现存上届控制在90%以内
config.gpu_options.allow_growth = True

config = tf.ConfigProto(allow_soft_placement = True)

sess = tf.Session(config = config)
#sess = tf.Session(config=config)
KTF.set_session(sess)

# 所有读出数据存储地址
specific_path = '。。'

log_filepath = specific_path

lr_reminder = []
ssum = 0
global_epoch = 0

# 寻找hard example（舍弃）
'''
class FindHardCase(Callback):
    def __init__(self):
        super(FindHardCase, self).__init__()
    def on_epoch_end(self, epoch, logs=None):


    def on_batch_end(self, batch, logs=None):
'''

# 三角学习率 callback，可做备选
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.

        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
        print("lr changed to {}".format(self.clr()))


def random_rotation(x, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


# 图像的随机平移（上下左右，输入为 [ Channel , H , W ] 三维张量）
def random_shift(x, wrg, hrg, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0.):
    h, w = x.shape[row_index], x.shape[col_index]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x

# 错切
def random_shear(x, intensity, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0.):
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x

# 放缩
def random_zoom(x, zoom_range, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


# 随机换轴（相当于特殊角度旋转）
def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


# 图像旋转  需要（1.坐标原点平移到中心 2.旋转 3.重置坐标原点）
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    # 平移-》旋转-》复位
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


# 把上述的变换矩阵实施到输入input上，input维数为3
def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    # 把channel放到0
    x = np.rollaxis(x, channel_index, 0)

    # 提取旋转要素
    final_affine_matrix = transform_matrix[:2, :2]
    # 提取平移要素
    final_offset = transform_matrix[:2, 2]

    # 逐通道处理（for x_channel in x）
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=1, mode=fill_mode, cval=cval) for x_channel
                      in x]
    # 拼接，与concatenate类似
    x = np.stack(channel_images, axis=0)
    # 把channel放到末尾
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


# 在axis轴上对图像进行反转flip
def flip_axis(x, axis):
    # 把需要反转的轴放到0
    x = np.asarray(x).swapaxes(axis, 0)
    # 0 轴翻转，其余保持
    x = x[::-1, ...]
    # 恢复原格式
    x = x.swapaxes(0, axis)

    return x


# 一维向量展开成图像
def array_to_img(x, dim_ordering='default', scale=True):
    from PIL import Image
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


# 图像压缩成一维向量
def img_to_array(img, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering: ', dim_ordering)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


# 图像读取
def load_img(path, grayscale=False, target_size=None, crop=(0, 0, 0)):

    from PIL import Image
    img = Image.open(path)
    w, h = img.size
    img = img.crop((crop[0], crop[1], w - crop[0], h - crop[1]))
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]), resample=Image.LANCZOS)
    original_size = w, h
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(root, f)
            for root, dirs, files in os.walk(directory) for f in files
            if re.match('([\w]+\.(?:' + ext + '))', f)]


class ImageDataGenerator(object):

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,

                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 depth_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 dim_ordering='default'):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering should be "tf" (channel after row and '
                             'column) or "th" (channel before row and column). '
                             'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        # 控制放缩尺度：（具体是否数据扩充还未知）
        if np.isscalar(zoom_range):  # 是否标量（非数组或字典之类的）
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]

        elif len(zoom_range) == 2:  # 若维度为2,则直接替代
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow_from_filenames_3d_class_train(self, X_filenames20, X_filenames30, X_filenames45, Y_classes=None, batch_size=32,crop20=(None, None, None),
                                           crop30=(None, None, None),crop45=(None, None, None), shuffle=True, seed=None,
                                            save_to_dir=None, save_prefix='', save_format='jpeg', shuttingdown_epoch=1000):
        return StackFileIteratorClass2(
            X_filenames20, X_filenames30, X_filenames45, Y_classes, self,
            batch_size=batch_size, crop20=crop20, crop30=crop30, crop45=crop45, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, border=shuttingdown_epoch)

    def standardize(self, x):
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = 0
        img_col_index = 1
        img_channel_index = 2


        if self.samplewise_center:
            x -= np.mean(x)
        if self.samplewise_std_normalization:
            x /= (np.std(x) + 1e-7)
        '''
        if self.samplewise_center and self.samplewise_std_normalization:
            mmax = np.max(x)
            mmin = np.min(x)
            x = (x - mmin) / (mmax - mmin)
        '''
        if self.featurewise_center:

            if self.mean is not None:
                x -= self.mean
            else:

                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')

        return x

    def random_transform(self, x):
        # x is a single image, so it doesn't have image number at index 0
        # 此函数只对单张图片进行变换！！！！

        img_row_index = 1
        img_col_index = 2  # [ Channel , Row , Col ]
        img_slice_index = 0

        # 1
        # use composition of homographies to generate final transform that needs to be applied
        # 采用单应变换的组合来实现 transform
        if self.rotation_range:  # default : 15
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
            # theta为旋转角度（通过均匀分布随机得到）
        else:
            theta = 0
        # 标准旋转矩阵：（行阵相乘）
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

        # 2
        # 平移范围（上下、左右）
        if self.height_shift_range:
            # （0-1）* 实际大小
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            # （0-1）* 实际大小
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        # 3
        # 错切范围：（错切：沿某一边界进行拉伸，不同于warp）（均匀分布随机）
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        # 4
        # 各向异性放缩
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        # 旋转 * 平移 * 错切 * 放缩（4位一体，当做一次‘ 特殊旋转‘ 处理）
        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]

        # 这里进入真正的旋转函数（ 平移 + 旋转族 + 复位 ）
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

        x = apply_transform(x, transform_matrix, img_slice_index,
                            fill_mode=self.fill_mode, cval=self.cval)

        # 随机通道偏移？？？
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_slice_index)

        # 对 col 的翻转表现为水平翻转
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)

        # 对 row 的翻转表现为垂直翻转
        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
        # 对 channel 的翻转表现为 depth 翻转
        if self.depth_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_slice_index)

        # TODO:
        # channel-wise normalization 基于通道的标准化
        # barrel 桶形失真（特殊情况）
        # fisheye 鱼眼失真（特殊情况）
        return x


class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)
        self.seed = seed

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                self.seed = seed + self.total_batches_seen
                np.random.seed(self.seed)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:  # 此句控制尾端不够一个batch的情况
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen = self.total_batches_seen + 1
            yield (index_array[current_index: current_index + current_batch_size], current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class StackFileIteratorClass2(Iterator):

    """ 生成图像流 ，注意训练时每个图像的衍生物要一一对应"""

    def __init__(self, X_filenames20, X_filenames30, X_filenames45, Y_classes, image_data_generator,
                 batch_size=32, crop20=(None, None, None), crop30=(None, None, None), crop45=(None, None, None), shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg', border=1000):

        self.dim_ordering = dim_ordering
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.crop20 = crop20
        self.crop30 = crop30
        self.crop45 = crop45
        global global_epoch
        self.current_epoch = global_epoch
        self.Y_classes = Y_classes
        self.batch_Filenames=[]
        self.batch_class=[]
        self.X_filenames20 = X_filenames20
        self.X_filenames30 = X_filenames30
        self.X_filenames45 = X_filenames45
        self.X_sample20 = np.load(X_filenames20[0])
        self.X_sample30 = np.load(X_filenames30[0])
        self.X_sample45 = np.load(X_filenames45[0])

        if Y_classes is not None:
            self.nb_class = len(np.unique(Y_classes))

        if Y_classes is not None and len(X_filenames30) != len(Y_classes):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' % (len(X_filenames30), len(Y_classes)))

        if self.X_sample30.ndim != 3:
            raise ValueError('loaded x data in `ScanStackFileIterator` '
                             'should have rank 3. Loaded array '
                             'has shape', self.X_sample30.shape)

        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.shuttingdown_border=border
        self.save_format = save_format
        super(StackFileIteratorClass2, self).__init__(len(X_filenames30), batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        self.batch_class.clear()
        batch_shape20 = [current_batch_size] + list(self.X_sample20.shape)[0:]
        batch_shape30 = [current_batch_size] + list(self.X_sample30.shape)[0:]
        batch_shape45 = [current_batch_size] + list(self.X_sample45.shape)[0:]
        global global_epoch
        self.current_epoch = global_epoch
        """ 提前将图像先裁剪边缘，扩充数据 """
        if self.crop20[0] is not None:
            batch_shape20[1] -= 2 * self.crop20[0]
            batch_shape20[2] -= 2 * self.crop20[1]
            batch_shape20[3] -= 2 * self.crop20[2]
        if self.crop30[0] is not None:
            batch_shape30[1] -= 2 * self.crop30[0]
            batch_shape30[2] -= 2 * self.crop30[1]
            batch_shape30[3] -= 2 * self.crop30[2]
        if self.crop45[0] is not None:
            batch_shape45[1] -= 2 * self.crop45[0]
            batch_shape45[2] -= 2 * self.crop45[1]
            batch_shape45[3] -= 2 * self.crop45[2]

        batch_x20 = np.zeros(tuple(batch_shape20))
        batch_x30 = np.zeros(tuple(batch_shape30))
        batch_x45 = np.zeros(tuple(batch_shape45))

        self.batch_Filenames.clear()
        if self.Y_classes is not None:

            batch_y = np.zeros((len(batch_x30), self.nb_class), dtype='float32')
            for i, label in enumerate(self.Y_classes[index_array]):
                batch_y[i, label] = 1.
            for j,k in enumerate(index_array):
                self.batch_class.append(self.Y_classes[k])

        for i, j in enumerate(index_array):
            x = np.load(self.X_filenames20[j])
            self.batch_Filenames.append(os.path.basename(self.X_filenames20[j]))
            np.random.seed(self.seed)
            if self.current_epoch < self.shuttingdown_border:
                x = self.image_data_generator.random_transform(x.astype('float32'))
                x = self.image_data_generator.standardize(x)
                # print('augmentation_used_on_epoch_{}--{}'.format(self.current_epoch,j))

            else:
                x = x.astype('float32')
                x = self.image_data_generator.standardize(x)
                # print('augmentation_shutting_down_on_epoch_{}--{}'.format(self.current_epoch,j))

            if self.crop20[0] is not None:
                bias=np.random.randint(self.crop20[0] + 1)
                x = x[self.crop20[0]-bias:-self.crop20[0]-bias, self.crop20[1]-bias:-self.crop20[1]-bias, self.crop20[2]-bias:-self.crop20[2]-bias, ...]

            batch_x20[i] = x

        for i, j in enumerate(index_array):
            x = np.load(self.X_filenames30[j])
            self.batch_Filenames.append(os.path.basename(self.X_filenames30[j]))
            np.random.seed(self.seed)
            if self.current_epoch < self.shuttingdown_border:
                x = self.image_data_generator.random_transform(x.astype('float32'))
                x = self.image_data_generator.standardize(x)
                # print('augmentation_used_on_epoch_{}--{}'.format(self.current_epoch,j))

            else:
                x = x.astype('float32')
                x = self.image_data_generator.standardize(x)
                # print('augmentation_shutting_down_on_epoch_{}--{}'.format(self.current_epoch,j))

            if self.crop30[0] is not None:
                bias=np.random.randint(self.crop30[0] + 1)
                x = x[self.crop30[0]-bias:-self.crop30[0]-bias, self.crop30[1]-bias:-self.crop30[1]-bias, self.crop30[2]-bias:-self.crop30[2]-bias, ...]

            batch_x30[i] = x

        for i, j in enumerate(index_array):
            x = np.load(self.X_filenames45[j])
            self.batch_Filenames.append(os.path.basename(self.X_filenames45[j]))
            np.random.seed(self.seed)
            if self.current_epoch < self.shuttingdown_border:
                x = self.image_data_generator.random_transform(x.astype('float32'))
                x = self.image_data_generator.standardize(x)
                # print('augmentation_used_on_epoch_{}--{}'.format(self.current_epoch,j))

            else:
                x = x.astype('float32')
                x = self.image_data_generator.standardize(x)
                # print('augmentation_shutting_down_on_epoch_{}--{}'.format(self.current_epoch,j))

            if self.crop45[0] is not None:
                bias2=np.random.randint(self.crop45[0] + 1)
                x = x[self.crop45[0]-bias2:-self.crop45[0]-bias2, self.crop45[1]-bias2:-self.crop45[1]-bias2, self.crop45[2]-bias2:-self.crop45[2]-bias2, ...]

            batch_x45[i] = x

        if self.dim_ordering == 'th':
            batch_x20 = np.expand_dims(batch_x20, 5).astype('float32')
            batch_x20 = np.swapaxes(np.rollaxis(batch_x20, -1), 0, 1)
            batch_x30 = np.expand_dims(batch_x30, 5).astype('float32')
            batch_x30 = np.swapaxes(np.rollaxis(batch_x30, -1), 0, 1)
            batch_x45 = np.expand_dims(batch_x45, 5).astype('float32')
            batch_x45 = np.swapaxes(np.rollaxis(batch_x45, -1), 0, 1)

            if self.Y_classes is not None:
                return batch_x30, batch_x30, batch_x45, batch_y

            else:
                return batch_x30, batch_x30, batch_x45

        batch_x20_out = np.expand_dims(batch_x20, 5).astype('float32')
        batch_x30_out = np.expand_dims(batch_x30, 5).astype('float32')
        batch_x45_out = np.expand_dims(batch_x45, 5).astype('float32')

        if self.Y_classes is not None:
            return batch_x20_out, batch_x30_out, batch_x45_out, batch_y, self.batch_Filenames, self.batch_class
        else:
            return batch_x20_out, batch_x30_out, batch_x45_out

class ModelCheckpointS3(Callback):
    """用于保存checkpoint（以 acc 或 loss 或 val_acc 或 val_loss 为测度）"""

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False, upload_to_s3=False,
                 mode='auto', period=1 ):
        super(ModelCheckpointS3, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.upload_to_s3 = upload_to_s3
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or 'dice' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            filepath = filepath+'_acc{0:.3f}_loss{1:.2f}_val_acc{2:.3f}_val_loss{3:.2f}———{4:d}.h5'.format(logs.get('acc'), logs.get('loss'), logs.get('val_acc'), logs.get('val_loss'), int(epoch))
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)

                        if self.upload_to_s3:
                            self.upload(filepath)
                            if self.verbose > 0:
                                print('upload of {} completed.'.format(os.path.basename(filepath)))


                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

                if self.upload_to_s3:
                    self.upload(filepath)
                    if self.verbose > 0:
                        print('upload of {} completed.'.format(os.path.basename(filepath)))

    @staticmethod
    def upload(source_path, chunk_size=50000000):
        creds = pd.DataFrame().from_csv('accessKeys.csv')
        connection = boto.connect_s3(creds['AWSAccessKeyId'][0], creds['AWSSecretKey'][0])
        bucket = connection.get_bucket('louisbucket')

        source_size = os.stat(source_path).st_size

        mp = bucket.initiate_multipart_upload(os.path.basename(source_path))

        chunk_count = int(math.ceil(source_size / float(chunk_size)))

        for i in range(chunk_count):
            offset = chunk_size * i
            bytes = min(chunk_size, source_size - offset)
            with FileChunkIO(source_path, 'r', offset=offset, bytes=bytes) as fp:
                mp.upload_part_from_file(fp, part_num=i + 1)

        # Finish the upload
        mp.complete_upload()

# 在训练中捕捉轮次，用于各种轮次有关的操作
class Cal_epoch(Callback):

    def __init__(self):
        super(Cal_epoch, self).__init__()
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        global global_epoch
        global_epoch = self.current_epoch

# 将训练数据和测试数据分为不同的 bag（20，30，45） ，每组内保证良恶性均衡并相同数据一一对应
def get_bags(ratio,  squ_list, ade_list, inf_list, ben_list, squ_label, ade_label, inf_label, ben_label,
             seed3=None, seed4=None,seed5=None, seed6=None,):

    patient_list_squ = copy.deepcopy(squ_list)
    patient_list_ade = copy.deepcopy(ade_list)
    patient_list_ben = copy.deepcopy(ben_list)
    patient_list_inf = copy.deepcopy(inf_list)
    bags = []
    # 训练集和测试集分裂比例
    split_ratio = ratio

    nbr_train1 = int(len(patient_list_squ) * split_ratio)
    nbr_train2 = int(len(patient_list_ben) * split_ratio)  ##ratio是0.95
    nbr_train3 = int(len(patient_list_ade) * split_ratio)
    nbr_train4 = int(len(patient_list_inf) * split_ratio)

    # 前90%用于训练
    train_patients_squ = patient_list_squ[:nbr_train1]
    train_patients_ben = patient_list_ben[:nbr_train2]
    train_patients_ade = patient_list_ade[:nbr_train3]
    train_patients_inf = patient_list_inf[:nbr_train4]

    train_label_squ = squ_label[:nbr_train1]
    train_label_ben = ben_label[:nbr_train2]
    train_label_ade = ade_label[:nbr_train3]
    train_label_inf = inf_label[:nbr_train4]

    # 后10%用于测试
    val_patients_squ = patient_list_squ[nbr_train1:]
    val_patients_ben = patient_list_ben[nbr_train2:]
    val_patients_ade = patient_list_ade[nbr_train3:]
    val_patients_inf = patient_list_inf[nbr_train4:]

    val_label_squ = squ_label[nbr_train1:]
    val_label_ben = ben_label[nbr_train2:]
    val_label_ade = ade_label[nbr_train3:]
    val_label_inf = inf_label[nbr_train4:]

    train_patients = train_patients_squ + train_patients_ben + train_patients_ade + train_patients_inf
    train_label = train_label_squ + train_label_ben + train_label_ade + train_label_inf

    val_patients = val_patients_squ + val_patients_ben + val_patients_ade + val_patients_inf
    val_label = val_label_squ + val_label_ben + val_label_ade + val_label_inf
    #####在融合网络中取消了seed3和seed4的打乱顺序操作

    if seed3 == None and seed4 == None and seed5 == None and seed6 == None:
        seed3 = np.random.randint(1002)
        seed4 = np.random.randint(1002)
        seed5 = np.random.randint(1002)
        seed6 = np.random.randint(1002)
    random.seed(seed3)
    random.shuffle(train_patients)
    random.seed(seed3)
    random.shuffle(train_label)
    random.seed(seed5)
    random.shuffle(train_patients)
    random.seed(seed5)
    random.shuffle(train_label)

    random.seed(seed4)
    random.shuffle(val_patients)
    random.seed(seed4)
    random.shuffle(val_label)
    random.seed(seed6)
    random.shuffle(val_patients)
    random.seed(seed6)
    random.shuffle(val_label)

    
    bags += [(train_patients, train_label, val_patients, val_label)]

    return bags, seed3, seed4,seed5,seed6

# 与 get_bags 的不同在于无视训练集和测试集的良恶性比例（可选测试）
def get_bags_random(ratio, folds, mag_list, ben_list, file_list, seed1 = None, seed2 = None, seed3 = None, seed4 = None):

    patient_list_malignancy = copy.deepcopy(mag_list)
    patient_list_benign = copy.deepcopy(ben_list)
    bags = []
    patient_list_sum = patient_list_malignancy + patient_list_benign
    # 训练集和测试集分裂比例
    split_ratio = ratio
    for i in range(folds):
        # 随机打乱
        if seed1==None and seed2==None:
            seed1 = np.random.randint(1002)
            seed2 = np.random.randint(1002)
        random.seed(seed1)
        random.shuffle(patient_list_sum)
        random.seed(seed2)
        random.shuffle(patient_list_sum)
        # 分裂数

        nbr_train1 = int(len(patient_list_sum) * split_ratio)
        # 前90%用于训练
        train_patients = patient_list_sum[:nbr_train1]

        # 后10%用于测试
        val_patients = patient_list_sum[nbr_train1:]


        if seed3==None and seed4==None:
            seed3 = np.random.randint(1002)
            seed4 = np.random.randint(1002)
        random.seed(seed3)
        random.shuffle(train_patients)
        random.seed(seed4)
        random.shuffle(val_patients)


        # 因为for循环，所以bag中有10组训练和测试组（10*[90%+10%]）
        # 在此处，list 由恶性和良性拼接而成。
        bags += [(train_patients, val_patients)]

    return bags, seed1, seed2, seed3, seed4

def main(arguments=None, squ_list20=None,
     ade_list20=None,
     inf_list20=None,
     ben_list20=None,

     squ_list30=None,
     ade_list30=None,
     inf_list30=None,
     ben_list30=None,

     squ_list45=None,
     ade_list45=None,
     inf_list45=None,
     ben_list45=None,

     squ_label=None,
     ade_label=None,
     inf_label = None,
     ben_label = None):

    n_epoch_0 = arguments.n_epochs
    batch_size = arguments.batch_size

    print('begin training, saving to weights/{}.h5, upload to s3 = {}'.format(arguments.weight_filename, arguments.s3))

    # 定义训练数据的操作（生成训练数据生成器）
    train_datagen = ImageDataGenerator(
        # 下面两项控制图像标准化（减均值除标准差）
        samplewise_center=True,
        samplewise_std_normalization=True,
        # 下面控制缩放比例（1-zoom，1+zoom）
        zoom_range=0.1,
        # 控制旋转范围
        rotation_range=arguments.rotation,
        # 错切范围
        shear_range=arguments.shear,
        # 平移范围（上下，左右）
        height_shift_range=0.2,
        width_shift_range=0.2,
        # 随机翻转（水平，垂直，前后）
        horizontal_flip=True,
        vertical_flip=True,
        depth_flip=True
    )
    # 定义测试数据的操作（生成测试数据生成器）
    val_datagen = ImageDataGenerator(  # 图像标准化
        samplewise_center=True,
        samplewise_std_normalization=True)

    bags30,seed3, seed4,seed5,seed6 = get_bags(ratio=0.89, #folds=arguments.total_folds,
                                    squ_list=squ_list30, squ_label=squ_label,
                                    ade_list=ade_list30,ade_label=ade_label,
                                    inf_list=inf_list30, inf_label=inf_label,
                                    ben_list=ben_list30, ben_label=ben_label
                                    )
    bags45 = get_bags(ratio=0.89,  squ_list=squ_list45,squ_label=squ_label, ade_list=ade_list45, ade_label=ade_label,inf_list=inf_list45,
                      inf_label=inf_label,ben_list=ben_list45, ben_label=ben_label,seed3=seed3, seed4=seed4,seed5=seed5,seed6=seed6)[0]
#如果是class（6），ratio就是0.8515，如果是class（duo）就是0.8509
    bags20 = get_bags(ratio=0.89, squ_list=squ_list20, squ_label=squ_label,ade_list=ade_list20,ade_label=ade_label,inf_list=inf_list20,
             inf_label=inf_label,ben_list=ben_list20, ben_label=ben_label,seed3=seed3, seed4=seed4,seed5=seed5,seed6=seed6)[0]

    print('bags20_shape:', np.shape(bags20))
    print('bags30_shape:', np.shape(bags30))
    print('bags45_shape:', np.shape(bags45))

    for n_fold,((train_patients20,train_label20, val_patients20,val_label20), (train_patients30, train_label30,val_patients30,val_label30),
                (train_patients45,train_label45, val_patients45,val_label45)) in enumerate(zip(bags20, bags30, bags45)):

        # 如果测试集和训练集有交集，则触发异常
        train_scanfile_list20 = train_patients20
        train_scanfile_list30 = train_patients30
        train_scanfile_list45 = train_patients45
        # train_scanfile_list.pop(0)
        # train_scanfile_list.pop(1)
        val_scanfile_list20 = val_patients20
        val_scanfile_list30 = val_patients30
        val_scanfile_list45 = val_patients45

        # 存储当前训练集和测试集为 txt
        file_train20 = open('...','w')
        file_test20 = open('...','w')

        file_train30 = open('...', 'w')
        file_test30 = open('...','w')
        file_train45 = open('...','w')
        file_test45 = open('...','w')


        for sub in train_patients20:

            file_train20.write(str(sub))
            file_train20.write('\n')
        file_train20.close()

        for sub in val_scanfile_list20:
            file_test20.write(str(sub))
            file_test20.write('\n')
        file_test20.close()

        for sub in train_patients30:

            file_train30.write(str(sub))
            file_train30.write('\n')
        file_train30.close()

        for sub in val_scanfile_list30:
            file_test30.write(str(sub))
            file_test30.write('\n')
        file_test30.close()

        for sub in train_patients45:
            file_train45.write(str(sub))
            file_train45.write('\n')
        file_train45.close()

        for sub in val_scanfile_list45:
            file_test45.write(str(sub))
            file_test45.write('\n')
        file_test45.close()

        #train_label=train_label20+train_label30+train_label45
        train_scanfile_truth = np.array(train_label30)
        #val_label=val_label20+val_label30+val_label45
        val_scanfile_truth = np.array(val_label30)

        # 同理，判定无交集
        assert (len(set(train_scanfile_list20).intersection(val_scanfile_list20)) == 0)
        assert (len(set(train_scanfile_list30).intersection(val_scanfile_list30)) == 0)
        assert (len(set(train_scanfile_list45).intersection(val_scanfile_list45)) == 0)

        print('train and val 分别有：',len(train_scanfile_list30), len(val_scanfile_list30))

        train_ratio = 0
        test_ratio = 0
        train_sum = sum(train_scanfile_truth)
        train_ratio = train_sum/len(train_scanfile_truth)
        test_sum = sum(val_scanfile_truth)
        test_ratio = test_sum / len(val_scanfile_truth)

        file_train_ratio = open('...','w')
        file_train_ratio.write(str(train_ratio))
        file_train_ratio.write('\n')
        file_train_ratio.close()

        file_test_ratio = open('...','w')
        file_test_ratio.write(str(test_ratio))
        file_test_ratio.write('\n')
        file_test_ratio.close()


        train_generator = train_datagen.flow_from_filenames_3d_class_train(train_scanfile_list20, train_scanfile_list30, train_scanfile_list45, train_scanfile_truth, crop20=(2, 2, 2), crop30=(2, 2, 2), crop45=(2, 2, 2),
                                                                     batch_size=batch_size, shuffle=True, seed=None, shuttingdown_epoch=arguments.shuttingdown_epoch)


        val_generator = val_datagen.flow_from_filenames_3d_class_train(val_scanfile_list20, val_scanfile_list30, val_scanfile_list45, val_scanfile_truth, crop20=(2, 2, 2), crop30=(2, 2, 2), crop45=(2, 2, 2),
                                                                 batch_size=batch_size, shuffle=True, seed=None, shuttingdown_epoch=arguments.shuttingdown_epoch)

        arguments.n_epochs = n_epoch_0
###下面是多分类的ROC曲线绘制过程###########
        #model_3d = DPSEN_Contruct(input_shape1=(26, 26, 26, 1), input_shape2=(41, 41, 41, 1), input_shape3=(16, 16, 16, 1), cardinality=1,ratio=4)
        
        '''
        
        # 用于已训练后网络计算 AUC

        ypred=model_3d.predict_generator(
            val_generator,
            len(val_scanfile_list45) // batch_size,
            workers=0,
            use_multiprocessing=False,
            max_queue_size=10)

        # ypred2 = np.argmax(ypred, axis=1)
        ypred2 = ypred[:, 1]
        yt = val_scanfile_truth1
        np.savetxt('C:\\Users\\Fizz\\Desktop\\ROC Save\\y_truth.txt',yt,fmt='%lf',delimiter=',')
        np.savetxt('C:\\Users\\Fizz\\Desktop\\ROC Save\\y_predict.txt', ypred2, fmt='%lf',delimiter=',')

        false_positive_rate, true_positive_rate, thresholds = roc_curve(yt, ypred2)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        plt.title('Receiver Operating Characteristic')

        xnew = np.linspace(-0.1, 1.2, 100)

        np.savetxt('C:\\Users\\Fizz\\Desktop\\ROC Save\\false_positive_rate.txt',false_positive_rate,fmt='%lf',delimiter=',')
        np.savetxt('C:\\Users\\Fizz\\Desktop\\ROC Save\\true_positive_rate.txt', true_positive_rate, fmt='%lf',delimiter=',')

        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.005, 1.005])
        plt.ylim([-0.005, 1.005])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        auc1 = roc_auc_score(yt, ypred2)
        print(auc1)
        '''

        '''
        xpred = model_3d.predict_generator(
            train_auc_generator,
            len(train_scanfile_list) // batch_size,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10)
        xpred2 = np.argmax(xpred, axis=1)
        xt = train_scanfile_truth
        auc2 = roc_auc_score(xt, xpred2)
        print(auc2)
        '''


        best_model_file = specific_path + "{}_fold{}_".format(arguments.weight_filename, 0)

        # 用于保存效果最好的 model
        best_model = ModelCheckpointS3(best_model_file, monitor='val_loss', verbose=1,
                                       save_best_only=True, upload_to_s3=arguments.s3)

        opti = SGD(lr=0.003, momentum=0.5, decay=0.0005, nesterov=True)
        #opti = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        with tf.device('/cpu:0'):

            model_3d_GPU = DPSEN_Contruct(input_shape1=(26, 26, 26, 1), input_shape2=(41, 41, 41, 1),
                                          input_shape3=(16, 16, 16, 1), cardinality=1, ratio=4)

        if arguments.using_parallel:
            model_3d = multi_gpu_model(model_3d_GPU, 2)
        else:
            model_3d = model_3d_GPU
        # 是否分GPU并行
        # if arguments.using_parallel:
        #     model_3d = multi_gpu_model(model_3d, 2)
        
        #model_3d.compile(loss='categorical_crossentropy', optimizer=opti, metrics=[tf.keras.metrics.categorical_accuracy])#['accuracy'])
        model_3d.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])#categorical_crossentropy,weighted_binary_crossentropy


        '''
        #load 测试 acc val_acc
    
        model_3d.load_weights('E:\\Fizz\\PyCharm Community Edition 2017.3.3\\Pycharm Project\\Best_weights.h5')

        print(model_3d.weights)# 获得模型所有权值
        t = model_3d.get_layer('conv3d_13')
        print(t)
        print(model_3d.get_weights()[2])

        test_eva = model_3d.evaluate_generator(
            val_generator,
            len(val_scanfile_list) // batch_size,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10)
        print(test_eva)


        train_eva = model_3d.evaluate_generator(
            train_generator,
            len(train_scanfile_list) // batch_size,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10)
        print(train_eva)
        '''


        def scheduler(epoch):
            # 每隔300个epoch，学习率减小为原来的0.7，并记录学习率

            if epoch % 150 == 0 and epoch != 0:
                lr = K.get_value(model_3d.optimizer.lr)
                K.set_value(model_3d.optimizer.lr, lr * 0.7)
                print("lr changed to {}".format(lr * 0.7))
                lr_reminder.append(lr * 0.7)
                if os.path.exists('...'):
                    os.remove('.')
                    file = open('...', 'w')
                    for fp in lr_reminder:
                        file.write(str(fp))
                        file.write('\n')

                else:
                    file = open('...', 'w')
                    for fp in lr_reminder:
                        file.write(str(fp))
                        file.write('\n')

            return K.get_value(model_3d.optimizer.lr)

        # 另一种学习率尝试方式
        def scheduler2(epoch):
            # 每隔30个epoch，学习率减小为原来的1/2
            global ssum

            if epoch == 0:
                ssum += 1
                lr = 0.001
                K.set_value(model_3d.optimizer.lr, lr)
                print("lr changed to {}".format(lr))
                lr_reminder.append(lr)

            if epoch == 10:
                K.set_value(model_3d.optimizer.lr, 0.0008)
                print("lr changed to {}".format(0.0008))
                lr_reminder.append(0.0008)

            if epoch == 35:
                K.set_value(model_3d.optimizer.lr, 0.0004)
                print("lr changed to {}".format(0.0004))
                lr_reminder.append(0.004)

            if epoch % 60 == 0 and epoch != 0:
                lr = K.get_value(model_3d.optimizer.lr)
                K.set_value(model_3d.optimizer.lr, lr * 0.1)
                print("lr changed to {}".format(lr * 0.1))
                lr_reminder.append(lr * 0.1)

                if os.path.exists('C:\\Users\\Desktop\\keras3\\lr_reminder_{}.txt'.format(ssum)):
                    os.remove('C:\\Users\\Desktop\\keras3\\lr_reminder_{}.txt'.format(ssum))
                    file = open('C:\\Users\\Desktop\\keras3\\lr_reminder_{}.txt'.format(ssum), 'w')
                    for fp in lr_reminder:
                        file.write(str(fp))
                        file.write('\n')

                else:
                    file = open('C:\\Users\\Desktop\\keras3\\lr_reminder_{}.txt'.format(ssum), 'w')
                    for fp in lr_reminder:
                        file.write(str(fp))
                        file.write('\n')

            return K.get_value(model_3d.optimizer.lr)

        lr_reducer = LearningRateScheduler(scheduler)
        earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
        epoch_cal=Cal_epoch()
        writer = TensorBoard(log_dir=log_filepath, write_graph=True, write_images=1, histogram_freq=0)
        csv_logger = CSVLogger(specific_path + 'Epoch--Acc--Loss--Lr--Val_acc--Val_loss{}'.format(0) + '.csv')

        #三角学习率（效果不佳）
        clr = CyclicLR(base_lr=0.0001, max_lr=0.01, step_size=25)

        # model_3d.load_weights()
        model_3d.fit_generator(train_generator, validation_data=val_generator,
                               validation_steps=len(val_scanfile_list30) // batch_size,
                               steps_per_epoch=len(train_scanfile_list30) // batch_size,
                               epochs=arguments.n_epochs, callbacks=[lr_reducer, best_model, csv_logger, writer, epoch_cal], verbose=1, workers=0) #, val_auc_generator = val_auc_generator, train_auc_generator=train_auc_generator





if __name__ == '__main__':

###路径需要修改
    data_path20_0 = r'...'
    data_path20_1 = r'...'
    data_path20_2 = r'...'
    data_path20_3 = r'...'

    data_path30_0 = r'...'
    data_path30_1 = r'...'
    data_path30_2 = r'...'
    data_path30_3 = r'...'

    data_path45_0 = r'...'
    data_path45_1 = r'...'
    data_path45_2 = r'...'
    data_path45_3 = r'...'



    xls_path0 = r'...'
    xls_path1 = r'...'
    xls_path2 = r'...'
    xls_path3 = r'...'
    

    def label_read(xls_path):
        table = xlrd.open_workbook(xls_path)
        temp = table.sheet_by_name('Sheet1')
        Excel_contain = {}
        for i in range(temp.ncols):
            temp1 = temp.col_values(i)
            Excel_contain[temp1[0]] = temp1[1:]

        total_truth = Excel_contain['mal']  ###########这地方需要修改
        for num0, ssub in enumerate(total_truth):
            total_truth[num0] = int(ssub)
        return total_truth

    inf_list20 = glob(data_path20_0 + '*.npy')  # 炎症
    squ_list20 = glob(data_path20_1 + '*.npy')  # 鳞癌
    ade_list20 = glob(data_path20_2 + '*.npy')  # 腺癌
    ben_list20 = glob(data_path20_3 + '*.npy')  # 其他

    inf_list30 = glob(data_path30_0 + '*.npy')  # 炎症
    squ_list30 = glob(data_path30_1 + '*.npy')  # 鳞癌
    ade_list30 = glob(data_path30_2 + '*.npy')  # 腺癌
    ben_list30 = glob(data_path30_3 + '*.npy')  # 其他

    inf_list45 = glob(data_path45_0 + '*.npy')  # 炎症
    squ_list45 = glob(data_path45_1 + '*.npy')  # 鳞癌
    ade_list45 = glob(data_path45_2 + '*.npy')  # 腺癌
    ben_list45 = glob(data_path45_3 + '*.npy')  # 其他

    inf_label = label_read(xls_path0)
    squ_label = label_read(xls_path1)
    ade_label = label_read(xls_path2)
    ben_label = label_read(xls_path3)

    print('squ_list: ', len(squ_list30))  # 打印有多少个squ类型的病例
    print('ade_list: ', len(ade_list30))
    print('inf_list:', len(inf_list30))
    print('ben_list: ', len(ben_list30))

    parser = argparse.ArgumentParser(description='Training Dual Path Squeeze and Excitation Network')
    parser.add_argument('-s3', action='store_true')
    parser.add_argument('--rotation', type=float, default=15.0)  
    parser.add_argument('--shear', type=float, default=15.0)#错切
    parser.add_argument('--weight_filename', type=str, default='weights')
    parser.add_argument('--total_folds', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--load_weights', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--shuttingdown_epoch', type=int, default=1300)  # 舍弃
    parser.add_argument('--using_parallel', type=bool, default=True)  # 默认不用

    args = parser.parse_args()

    main(arguments=args,
     squ_list20=squ_list20,
     ade_list20=ade_list20,
     inf_list20=inf_list20,
     ben_list20=ben_list20,

     squ_list30=squ_list30,
     ade_list30=ade_list30,
     inf_list30=inf_list30,
     ben_list30=ben_list30,

     squ_list45=squ_list45,
     ade_list45=ade_list45,
     inf_list45=inf_list45,
     ben_list45=ben_list45,

     squ_label=squ_label,
     ade_label=ade_label,
     inf_label = inf_label,
     ben_label = ben_label

     )