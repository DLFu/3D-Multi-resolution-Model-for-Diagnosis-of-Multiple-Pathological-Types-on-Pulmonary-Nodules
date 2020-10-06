# coding=utf-8
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.layers import Conv3D
from keras.layers import GlobalMaxPooling3D, Concatenate
from keras.layers import GlobalAveragePooling3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from glob import glob
import tqdm
import numpy as np
from keras.models import Model
from keras.layers import Input,Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers import Multiply, multiply
from keras.layers import add,Add,Average
from keras.regularizers import l2
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.backend import int_shape
from keras.backend import reshape
from keras.backend import mean
from keras.backend import var
from keras.backend import cast
from keras.backend import sqrt
import tensorflow as tf
from keras.utils import plot_model
from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.normalization import BatchNormalization

__all__ = ['DPSEN_Contruct', 'DPSEN','preprocess_input', 'decode_predictions']

class GroupNorm(Layer):

    """Group normalization layer 需自定义"""

    def __init__(self, groups=8, axis=-1, epsilon=1e-5, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones', beta_regularizer=None,
                 gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs):

        super(GroupNorm, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center  # gamma是否起作用
        self.scale = scale  # beta是否起作用

        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)

        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)

        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        """
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '                       
                                                        'input tensor should have a defined dimension '             
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        if dim < self.groups:
                raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '        
                                                                           'more than the number of channels (' + str(dim) + ').')
        if dim % self.groups != 0:
                    raise ValueError('Number of groups (' + str(self.groups) + ') must be a '                                   
                                                                               'multiple of the number of channels (' + str(dim) + ').')
        """
        self.input_spec = InputSpec(ndim=len(input_shape), axes={self.axis: dim})
        shape = (dim,)
        if self.scale:
            self.gamma = self.add_weight(shape=shape, name='gamma', initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer, constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape, name='beta', initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer, constraint=self.beta_constraint)
        else:
            self.beta = None
        super(GroupNorm, self).build(input_shape)
        self.built = True


    def call(self, inputs, **kwargs):

        input_shape = K.int_shape(inputs)  # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        reshape_group_shape = list(input_shape)
        reshape_group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape = [-1, self.groups]
        group_shape.extend(reshape_group_shape[1:])
        group_reduction_axes = list(range(len(group_shape)))
        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])                         # 维度不匹配时需要采用 broadcasting 机制（numpy）

        inputs = K.reshape(inputs, group_shape)
        mean = K.mean(inputs, axis=group_reduction_axes[2:], keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes[2:], keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        original_shape = [-1] + list(input_shape[1:])
        inputs = K.reshape(inputs, original_shape)

        if needs_broadcasting:
            outputs = inputs
            # In this case we must explicitly broadcast all parameters.
            if self.scale:
                broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                outputs = outputs * broadcast_gamma
            if self.center:
                broadcast_beta = K.reshape(self.beta, broadcast_shape)
                outputs = outputs + broadcast_beta
        else:
            outputs = inputs
            if self.scale:
                outputs = outputs * self.gamma
            if self.center:
                outputs = outputs + self.beta
        return outputs

    def get_config(self):
        config = {'groups': self.groups, 'axis': self.axis,
                  'epsilon': self.epsilon, 'center': self.center, 'scale': self.scale,
                  'beta_initializer': initializers.serialize(self.beta_initializer),
                  'gamma_initializer': initializers.serialize(self.gamma_initializer),
                  'beta_regularizer': regularizers.serialize(self.beta_regularizer),
                  'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'beta_constraint': constraints.serialize(self.beta_constraint),
                  'gamma_constraint': constraints.serialize(self.gamma_constraint)}

        base_config = super(GroupNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


'''
def GroupNorm1(x, G=9, eps=1e-5):  # 3 维输入的Group Norm（ N,H,W,L,C ）
    N, H, W, L, C = x.shape
    x = reshape(x, [cast(N, tf.int32), cast(H, tf.int32), cast(W, tf.int32), cast(L, tf.int32), cast(G, tf.int32),
                    cast(C // G, tf.int32)])
    mean_val = mean(x, axis=[1, 2, 3, 4], keepdims=True)

    var_val = var(x, axis=[1, 2, 3, 4], keepdims=True)

    # mean,var=tf.nn.moments(x,[1,2,3,4],keep_dims=True)
    x = (x - mean_val) / sqrt(var_val + eps)

    x = reshape(x, [cast(N, tf.int32), cast(H, tf.int32), cast(W, tf.int32), cast(L, tf.int32), cast(C, tf.int32)])

    gamma = K.ones(shape=(1, 1, 1, 1, cast(C, tf.int32)), name="gamma")
    beta = K.zeros(shape=(1, 1, 1, 1, cast(C, tf.int32)), name="beta")
    return x * gamma + beta
'''

def preprocess_input(x, data_format=None):

    """ 输入预处理,减均值除标准差"""

    if data_format is None:
        data_format = K.image_data_format()  # channel first 或 channel last
    assert data_format in {'channels_last', 'channels_first'}  # 断言语句，用于检测（相当于try catch）

    '''
    if data_format == 'channels_first':
        # 'RGB'->'BGR'                                           # 为了兼容opencv 需要转换为GBR
        x = x[:, ::-1, :, :,  : ]                                     # 以channel为基准倒序复制一遍
        #     N   C   H   W   L

        # Zero-center by mean pixel                              # 针对Imagenet的处理，标准化

        x[:, 0, :, :, :] -= 104                                     # 减均值
        x[:, 1, :, :, :] -= 117
        x[:, 2, :, :, :] -= 128
    else:
        # 'RGB'->'BGR'
        x = x[:, :, :, : ::-1]

        # Zero-center by mean pixel
        x[:, :, :, :, 0] -= 104
        x[:, :, :, :, 1] -= 117
        x[:, :, :, :, 2] -= 124

    x *= 0.0167                                                   # 除以标准差
    '''
    return x


def DPSEN_Contruct(input_shape1=None,
                    input_shape2=None,
                    input_shape3=None,
                    initial_conv_filters=64,
                    depth=[2,1,1],#241
                    filter_increment=[16, 16, 16],
                    cardinality=1,
                    width=3,
                    weight_decay=0,
                    include_top=True,
                    weights=None,
                    pooling=None,
                    classes=4,
                    test=0,
                    #test=1,
                    layer_name='input_1',
                    ratio=9):

    """ 利用_Create_DPSEN构建完整的输入输出 MODEL ；通路数可选（1，2，3）；test参数用于保存网络验证"""

    assert len(depth) == len(filter_increment), "The length of filter increment list must match the length of the depth list."


    img_input1 = Input(shape=input_shape1, name='input1')  # 转换为keras tensor
    if input_shape2:
        img_input2 = Input(shape=input_shape2, name='input2')
    if input_shape3:
        img_input3 = Input(shape=input_shape3,name='input3')


    # print(int_shape(img_input))
    x = _Create_DPSEN(nb_classes=classes, img_input=img_input1, include_top=include_top, initial_conv_filters=initial_conv_filters,
                    filter_increment=filter_increment, depth=depth, cardinality=cardinality, width=width, weight_decay=weight_decay, pooling=pooling,ratio=ratio)
    if input_shape2:
        x2 = _Create_DPSEN(nb_classes=classes, img_input=img_input2, include_top=include_top, initial_conv_filters=initial_conv_filters,
                    filter_increment=filter_increment, depth=depth, cardinality=cardinality, width=width, weight_decay=weight_decay, pooling=pooling,ratio=ratio)
    if input_shape3:
        x3 = _Create_DPSEN(nb_classes=classes, img_input=img_input3, include_top=include_top, initial_conv_filters=initial_conv_filters,
                    filter_increment=filter_increment, depth=depth, cardinality=cardinality, width=width, weight_decay=weight_decay, pooling=pooling,ratio=ratio)


    if not input_shape2 and not input_shape3:

        model = Model(inputs=img_input1, outputs=x)  # 创建model(单模型model)

    elif not input_shape3:

        added = Average()([x, x2])
        model = Model(inputs=[img_input1, img_input2], outputs=added)  # 创建model
    else:
        added = Average()([x, x2, x3])
        #added=Average(activation)[x,x2,x3]
        model = Model(inputs=[img_input1, img_input2, img_input3], outputs=added)  # 创建model


    if test:#这个没用上。后期测试用的还是训练里面额参数，这个没被调用
        model.load_weights('。。。')
        layer_name0 = layer_name
        intermediate_layer_model = Model(input=img_input1, outputs=model.get_layer(layer_name0).output)
        return intermediate_layer_model
    else:
        return model


def DPSEN(input_shape1=None,
          input_shape2=None,
          input_shape3=None,
          include_top=True,
          weights=None,
          pooling=None,
          cardinality=1,#控制resnext，分组卷积的残差网络，后期因为容易错乱，所以也没用
          classes=4,
          test=0,
          layer_name='input_1',
          ratio=9):#
    return DPSEN_Contruct(input_shape1=input_shape1, input_shape2=input_shape2, input_shape3=input_shape3, include_top=include_top, weights=weights,
                           pooling=pooling, classes=classes,test=test,layer_name=layer_name,cardinality=cardinality,ratio=ratio)



# momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
# moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
# 第一层初始化（45 -> 23 -> 12）
def _Initial_Conv3d_GN_ReLu_Block(input, initial_conv_filters, weight_decay=0.01):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
   # print("Init:", int_shape(input))
    x = Conv3D(initial_conv_filters, (5, 5, 5), padding='same', use_bias=False, kernel_initializer='glorot_normal',#可选替换glorot_normal，he_normal
               kernel_regularizer=l2(weight_decay),strides=(2, 2, 2))(input)  # 第一次降大小
   # print("After Conv:", int_shape(x))
    x = GroupNorm(axis=channel_axis)(x)
    #x = BatchNormalization(axis=1)(x) # 轴的位置需要尝试换算
    # print("After GN:", int_shape(x))
    #x = Dropout(0.25)(x)
    x = Activation('tanh')(x)#可选替换LRelu
    #x = LeakyReLU()(x)
    x = MaxPooling3D((5, 5, 5), strides=(2, 2, 2), padding='same')(x)

    #x = AveragePooling3D((5, 5, 5), strides=(2, 2, 2), padding='same')(x)
    return x


# Conv3D + BN + ReLu
def _GN_ReLu_Conv3d_Block(input, filters, kernel=(3, 3, 3), stride=(1, 1, 1), weight_decay=0.01):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # 默认是channel last
    x = Conv3D(filters, kernel, padding='same', use_bias=False, kernel_initializer='glorot_normal',#可选替换glorot_normal，he_normal
                   kernel_regularizer=l2(weight_decay),    strides=stride)(input)
    x = GroupNorm(axis=channel_axis)(x)
    #x = BatchNormalization(axis=1)(x)
    #x=Dropout(0.15)(x)
    x = Activation('tanh')(x)#可选替换LRelu
    #x = LeakyReLU()(x)



    return x


# 实现 Group 操作中的 3*3*3 模块，若cardinality为1 则不采用（1.group convolution 2.第一次卷积正常）
def _Grouped_Conv3D_GN_ReLu_Block(input, grouped_channels, cardinality, strides, weight_decay=0.01):#之前是-4

    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution          conv + BN + ReLu
        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=strides,
                   kernel_regularizer=l2(weight_decay), kernel_initializer='glorot_normal')(input)
        x = GroupNorm(axis=channel_axis)(x)
        #x = BatchNormalization(axis=1)(x)
        x = Activation('tanh')(x)
        #x = LeakyReLU()(x)

        #x=Dropout(0.3)(x)

        return x

    for c in range(cardinality):  # 取一个group做卷积，并concat加入list
        # x = GroupNorm(axis=channel_axis)(input)
        # #x = BatchNormalization(axis=1)(x)
        # #x = Activation('relu')(x)
        # x = LeakyReLU()(x)
        # #x = Dropout(0.3)(x)

        x = Lambda(lambda z: z[:, :, :, :, c * grouped_channels:(c + 1) * grouped_channels]  # lambda表达式定义函数或变量。
        if K.image_data_format() == 'channels_last' else
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :, :])(x)

        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=strides,
                   kernel_regularizer=l2(weight_decay),kernel_initializer='glorot_normal')(x)#

        group_list.append(x)
    group_merge = concatenate(group_list, axis=channel_axis)  # 给定channel轴，将通道再合并回来。
    #group_merge = BatchNormalization(axis=channel_axis)(group_merge)
    group_merge = GroupNorm(axis=channel_axis)(group_merge)
    # group_merge= Dropout(0.2)(group_merge)
    group_merge = Activation('tanh')(group_merge)
    #group_merge = LeakyReLU()(group_merge)
    # group_merge= MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')( group_merge)

    return group_merge


# 创建DPSEN 网络单元
def _Dual_Path_Squeeze_And_Excitation_Block(input, pointwise_filters_a, grouped_conv_filters_b, pointwise_filters_c,
                     filter_increment, cardinality, block_type='normal', ratio=9):#之前ratio是9，现在改成6试试


    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    grouped_channels = int(grouped_conv_filters_b / cardinality)

    # 把上一层的channel N 等分（cardinality），得到独立的几个卷积团体，最后再concat回去。
    init = concatenate(input, axis=channel_axis) if isinstance(input,list) else input  # 如果输入是list，则concat到一起，否则就直接赋input

    # 根据block_type有不同的用法
    if block_type == 'projection':  # 第一层分裂时用，用于产生 res 和 dense 双通道。
        stride = (1, 1, 1)
        projection = True
    elif block_type == 'downsample':  # 强行规范化
        stride = (2, 2, 2)            # 之所以称之为 downsample 是因为步长为 2 ，以金字塔层级来看相当于降采样。
        projection = True
    elif block_type == 'normal':  # 正常的 res 点加层和 dense 叠加层通路。
        stride = (1, 1, 1)
        projection = False
    else:
        raise ValueError('`block_type` must be one of ["projection", "downsample", "normal"]. Given %s' % block_type)

    if projection:  # 手动撕裂! 用于第一层的res和dense的初始化（产生旧的上层信息）

        projection_path = _GN_ReLu_Conv3d_Block(init, filters=pointwise_filters_c + 2 * filter_increment,
                                                kernel=(1, 1, 1), stride=stride)
        input_residual_path = Lambda(lambda z: z[:, :, :, :, :pointwise_filters_c]
        if K.image_data_format() == 'channels_last' else
        z[:, :pointwise_filters_c, :, :, :])(projection_path)
        input_dense_path = Lambda(lambda z: z[:, :, :, :, pointwise_filters_c:]
        if K.image_data_format() == 'channels_last' else
        z[:, pointwise_filters_c:, :, :, :])(projection_path)
    else:
        input_residual_path = input[0]
        input_dense_path = input[1]
    # 1*1
    x = _GN_ReLu_Conv3d_Block(init, filters=pointwise_filters_a, kernel=(1, 1, 1))
    # 3*3（group conv ）
    x = _Grouped_Conv3D_GN_ReLu_Block(x, grouped_channels=grouped_channels, cardinality=cardinality, strides=stride)
    # 1*1（channel 增加了）
    x = _GN_ReLu_Conv3d_Block(x, filters=pointwise_filters_c + filter_increment,
                              kernel=(1, 1, 1))  # 这一层要增加卷积核个数（channel数）
    
    # 取x的前c个channel
    output_residual_path = Lambda(lambda z: z[:, :, :, :, :pointwise_filters_c]
    if K.image_data_format() == 'channels_last' else
    z[:, :pointwise_filters_c, :, :, :])(x)
    # 取x的c之后的channel（对应filter_increment）
    output_dense_path = Lambda(lambda z: z[:, :, :, :, pointwise_filters_c:]
    if K.image_data_format() == 'channels_last' else
    z[:, pointwise_filters_c:, :, :, :])(x)
    # 残差网络的通路
    residual_path = add([input_residual_path, output_residual_path])

    # 密度网络的通路（以channel为基准拼接）
    dense_path = concatenate([input_dense_path, output_dense_path], axis=channel_axis)

    # 返回的是一个list
    return [residual_path, dense_path]

    # 创建整个DPSEN BLOCK


#################################################################################################################################

def _Create_DPSEN(nb_classes, img_input, include_top, initial_conv_filters,
                filter_increment, depth, cardinality=1,  width=3, weight_decay=0.01, pooling=None,ratio=9):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    N = list(depth)

    base_filters = 128

    # block 1 (initial conv block)  初始层
    x = _Initial_Conv3d_GN_ReLu_Block(img_input, initial_conv_filters, weight_decay)
  #  print("Block1 initialize layer")

    # block 2 (projection block)    分裂层(包含第一层分裂层projection和之后的正常层normal）
    filter_inc = filter_increment[0]
    filters = int(cardinality * width * 24)
    '''
    print("cardinality:%d" % cardinality)
    print("width:%d" % width)

    print('Block2--projection layer')
    print("Before projection ")
    print(int_shape(x))
    '''
    x = _Dual_Path_Squeeze_And_Excitation_Block(x, pointwise_filters_a=filters,
                         grouped_conv_filters_b=filters,
                         pointwise_filters_c=base_filters,
                         filter_increment=filter_inc,
                         cardinality=cardinality,
                         block_type='projection',
                         ratio=ratio)


    for i in range(N[0] - 1):

        x = _Dual_Path_Squeeze_And_Excitation_Block(x, pointwise_filters_a=filters,
                             grouped_conv_filters_b=filters,
                             pointwise_filters_c=base_filters,
                             filter_increment=filter_inc,
                             cardinality=cardinality,
                             block_type='normal',
                             ratio=ratio)

    # remaining blocks
    for k in range(1, len(N)):

        filter_inc = filter_increment[k]  # 每层增加的 filter
        filters *= 2  # 256->512->1024->2048
        base_filters *= 2

        x = _Dual_Path_Squeeze_And_Excitation_Block(x, pointwise_filters_a=filters,
                             grouped_conv_filters_b=filters,
                             pointwise_filters_c=base_filters,
                             filter_increment=filter_inc,
                             cardinality=cardinality,
                             block_type='downsample',
                             ratio=ratio)


        for i in range(N[k] - 1):

            x = _Dual_Path_Squeeze_And_Excitation_Block(x, pointwise_filters_a=filters,
                                 grouped_conv_filters_b=filters,
                                 pointwise_filters_c=base_filters,
                                 filter_increment=filter_inc,
                                 cardinality=cardinality,
                                 block_type='normal',
                                 ratio=ratio)

    x = concatenate(x, axis=channel_axis)

    # # x = GroupNorm(axis=channel_axis)(x)
    # x = BatchNormalization(axis=1 )(x)
    # #x = Activation('tanh')(x)
    # x = MaxPooling3D()(x)
    # #x = GlobalAveragePooling3D()(x)
    # x = Flatten()(x)###lecun_normal---stddev=sqrt(1/fan_in),He----stddev = sqrt(2 / fan_in)
    # # x = Dense(2048, use_bias=False, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal',activation='sigmoid')(x)
    # # x = Dropout(0.5)(x)#http://www.caffecn.cn/?/question/1030尝试使用全局的average pooling来代替FC层
    # #x = Dense(128, use_bias=False, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal',activation='sigmoid')(x)
    # #x = Dropout(0.5)(x)
    # x = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),kernel_initializer='glorot_normal',#'lecun_normal','he_normal'，’he_normal‘ activation='sigmoid'
    #          activation='sigmoid')(x)

    if include_top:  
        avg = GlobalAveragePooling3D()(x)
        max = GlobalMaxPooling3D()(x)
        x = add([avg, max])
        x = Lambda(lambda z: 0.5 * z)(x)
        # x = Flatten()(x)
        x = Dense(256, use_bias=False, kernel_regularizer=l2(weight_decay), kernel_initializer='glorot_normal',#可选替换glorot_normal，he_normal
                  # activity_regularizer=regularizers.l1(0.01),
                  bias_regularizer=l2(0), activation='tanh')(x) #relu 或者lrelu
        x = Dropout(0.5)(x)
        x = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay), kernel_initializer='glorot_normal',
                  # activity_regularizer=regularizers.l1(0.01),
        bias_regularizer=l2(0), activation='softmax')(x)  # 
        #           分类数         偏置项                    卷积核正则                      卷积核初始方式
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling3D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling3D()(x)
        elif pooling == 'max-avg':
            a = GlobalMaxPooling3D()(x)
            b = GlobalAveragePooling3D()(x)
            x = add([a, b])
            x = Lambda(lambda z: 0.5 * z)(x)

        # x = K.softmax(x)  # 最后输出层利用 softmax 回归，得到的 tensor 与 one-hot 编码后的 label 对应
        #
    # return x

    return x


if __name__ == '__main__':

    # model_3d = DPSEN(input_shape1=(30, 30, 30, 1), input_shape2=(45, 45, 45, 1),input_shape3=(20, 20, 20, 1), cardinality=1,ratio=4)
    model_3d = DPSEN(input_shape1=(26, 26, 26, 1), input_shape2=(41, 41, 41, 1), input_shape3=(16, 16, 16, 1),cardinality=1, ratio=4)
    # writer = tf.summary.FileWriter("   ", tf.get_default_graph())
    # writer.close()
    model_3d.summary()
    plot_model(model_3d, show_shapes=True, to_file=r'   ')

