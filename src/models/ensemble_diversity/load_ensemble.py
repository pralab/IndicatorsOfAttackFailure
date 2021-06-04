from __future__ import print_function

import os

import tensorflow as tf
from secml.array import CArray
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml import CNormalizerMeanStd
from secml.utils import fm
from secml.utils.download_utils import dl_file
from tensorflow.keras.layers import AveragePooling2D, Flatten
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from src.models.ensemble_diversity.keras_wrapper import CClassifierKeras


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input, depth, num_classes=10, dataset='cifar10'):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = input
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    if dataset == 'mnist':
        poolsize = 7
    else:
        poolsize = 8
    x = AveragePooling2D(pool_size=poolsize)(x)
    final_features = Flatten()(x)
    logits = Dense(
        num_classes, kernel_initializer='he_normal')(final_features)
    outputs = Activation('softmax')(logits)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model, inputs, outputs, logits, final_features


adp_url = "http://ml.cs.tsinghua.edu.cn/~tianyu/ADP/pretrained_models/ADP_standard_3networks/cifar10_ResNet20v1_model" \
          ".159.h5"
adp_file = "adp_cifar10_ResNet20v1_alpha-2.0_beta-0.5_epoch-159.h5"
adp_advt_url = "http://ml.cs.tsinghua.edu.cn/~tianyu/ADP/pretrained_models/ADP_with_PGDtrain_3networks" \
               "/cifar10_ResNet20v1_model.124.h5"
adp_advt_file = "adp_cifar10_ResNet20v1_alpha-2.0_beta-0.5_epoch-124_advt-PGD.h5"


def load_model(adv_training=False):
    model_file = adp_advt_file if adv_training else adp_file
    model_url = adp_advt_url if adv_training else adp_url
    path = os.path.join(os.path.dirname(__file__), model_file)
    if not fm.file_exist(path):
        out_path = dl_file(url=model_url, output_dir=os.path.dirname(__file__))
        os.rename(out_path, path)

    n = 3
    depth = n * 6 + 2
    input_shape = (32, 32, 3)

    model_input = Input(shape=input_shape)
    model_dic = {}
    model_out = []
    for i in range(3):
        model_dic[str(i)] = resnet_v1(input=model_input, depth=depth)
        model_out.append(model_dic[str(i)][2])
    model_output = tf.keras.layers.concatenate(model_out)
    model = Model(inputs=model_input, outputs=model_output)
    model_ensemble = tf.keras.layers.Average()(model_out)
    model_ensemble = Model(inputs=model_input, outputs=model_ensemble)
    model.load_weights(path)
    tr, _ = CDataLoaderCIFAR10().load()
    tr = reshape_cifar10(tr)
    tr.X /= 255.
    tr_mean = tr.X.mean(axis=0)
    preprocess = CNormalizerMeanStd(
        mean=tr_mean.ravel(), std=CArray.ones(shape=tr_mean.shape[-1]))
    return CClassifierKeras(model=model_ensemble, pretrained=True,
                            input_shape=input_shape, preprocess=preprocess)


def reshape_cifar10(ds):
    x = ds.X.tondarray()
    x = x.reshape((x.shape[0], 3, 32, 32)).transpose((0, 2, 3, 1)).reshape(
        (x.shape[0], -1))
    ds.X = CArray(x)
    return ds
