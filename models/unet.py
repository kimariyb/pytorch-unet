# -*- conding: utf-8 -*-
"""
unet.py
This file is part of unet. This is a Main function file!

@author:
Kimariyb (kimariyb@163.com)

@license:
Licensed under the MIT License
For details, see the License file.

@Data:
2024/4/27 21:55
"""

from keras import layers, models


def encoder(img_input, activation='relu'):
    """
    Encode part of the U-Net model.
    
    Parameters
    ----------
    img_input : keras.layers.Input
        Input layer of the model.
    activation : str
        Activation function used in the model.
    
    Returns
    -------
    feat1, feat2, feat3, feat4, feat5 : keras.layers.Layer
        Output layers of the model.
    """
    # Block 1
    # Input size: 512x512x3
    # Output size: 512x512x64
    x = layers.Conv2D(
        64, (3, 3), activation=activation, padding='same', name='block1_conv1'
    )(img_input)
    x = layers.Conv2D(
        64, (3, 3), activation=activation, padding='same', name='block1_conv2'
    )(x)
    feat1 = x
    # MaxPooling
    # Input size: 512x512x64
    # Output size: 256x256x64
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    # Input size: 256x256x64
    # Output size: 256x256x128
    x = layers.Conv2D(
        128, (3, 3), activation=activation, padding='same', name='block2_conv1'
    )(x)
    x = layers.Conv2D(
        128, (3, 3), activation=activation, padding='same', name='block2_conv2'
    )(x)
    feat2 = x
    # MaxPooling
    # Input size: 256x256x128
    # Output size: 128x128x128
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    # Input size: 128x128x128
    # Output size: 128x128x256
    x = layers.Conv2D(
        256, (3, 3), activation=activation, padding='same', name='block3_conv1'
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation=activation, padding='same', name='block3_conv2'
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation=activation, padding='same', name='block3_conv3'
    )(x)
    feat3 = x
    # MaxPooling
    # Input size: 128x128x256
    # Output size: 64x64x256
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    # Input size: 64x64x256
    # Output size: 64x64x512
    x = layers.Conv2D(
        512, (3, 3), activation=activation, padding='same', name='block4_conv1'
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation=activation, padding='same', name='block4_conv2'
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation=activation, padding='same', name='block4_conv3'
    )(x)
    feat4 = x
    # MaxPooling
    # Input size: 64x64x512
    # Output size: 32x32x512
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # Input size: 32x32x512
    # Output size: 32x32x512
    x = layers.Conv2D(
        512, (3, 3), activation=activation, padding='same', name='block5_conv1'
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation=activation, padding='same', name='block5_conv2'
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation=activation, padding='same', name='block5_conv3'
    )(x)
    feat5 = x

    return feat1, feat2, feat3, feat4, feat5


def decoder(input_shape=(256, 256, 3), num_classes=21, activation='relu'):
    """
    Decode part of the U-Net model.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of the input image.
    num_classes : int    
        Number of classes in the dataset.
    activation : str
        Activation function used in the model.
    
    Returns
    -------
    output : keras.layers.Layer
        Output layer of the model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder part, Feautre maps
    feat1, feat2, feat3, feat4, feat5 = encoder(inputs)
    
    # Up-sampling
    # Input size: 32x32x512
    # Output size: 64x64x512
    P5_up = layers.UpSampling2D(size=(2, 2))(feat5)
    # Concatenate
    # Input size: 64x64x512
    # Output size: 64x64x1024
    P4 = layers.Concatenate(axis=3)([feat4, P5_up])
    # Convolutional layer
    # Input size: 64x64x1024
    # Output size: 64x64x512
    P4 = layers.Conv2D(
        512, 3, activation=activation, padding='same', kernel_initializer='he_normal'
    )(P4)
    P4 = layers.Conv2D(
        512, 3, activation=activation, padding='same', kernel_initializer='he_normal'
    )(P4)

    # Up-sampling
    # Input size: 64x64x512
    # Output size: 128x128x512
    P4_up = layers.UpSampling2D(size=(2, 2))(P4)
    # Concatenate
    # Input size: 128x128x512
    # Output size: 128x128x728
    P3 = layers.Concatenate(axis=3)([feat3, P4_up])
    # Convolutional layer
    # Input size: 128x128x728
    # Output size: 128x128x256
    P3 = layers.Conv2D(
        256, 3, activation=activation, padding='same', kernel_initializer='he_normal'
    )(P3)
    P3 = layers.Conv2D(
        256, 3, activation=activation, padding='same', kernel_initializer='he_normal'
    )(P3)
    
    # Up-sampling
    # Input size: 128x128x256
    # Output size: 256x256x256
    P3_up = layers.UpSampling2D(size=(2, 2))(P3)
    # Concatenate
    # Input size: 256x256x256
    # Output size: 256x256x384
    P2 = layers.Concatenate(axis=3)([feat2, P3_up])
    # Convolutional layer
    # Input size: 256x256x384
    # Output size: 256x256x128
    P2 = layers.Conv2D(
        128, 3, activation=activation, padding='same', kernel_initializer='he_normal'
    )(P2)
    P2 = layers.Conv2D(
        128, 3, activation=activation, padding='same', kernel_initializer='he_normal'
    )(P2)
    
    # Up-sampling
    # Input size: 256x256x128
    # Output size: 512x512x128
    P2_up = layers.UpSampling2D(size=(2, 2))(P2)
    # Concatenate
    # Input size: 512x512x128
    # Output size: 512x512x192
    P1 = layers.Concatenate(axis=3)([feat1, P2_up])
    # Convolutional layer
    # Input size: 512x512x192
    # Output size: 512x512x64
    P1 = layers.Conv2D(
        64, 3, activation=activation, padding='same', kernel_initializer='he_normal'
    )(P1)
    P1 = layers.Conv2D(
        64, 3, activation=activation, padding='same', kernel_initializer='he_normal'
    )(P1)
    
    # Output layer
    # Input size: 512x512x64
    # Output size: 512x512xnum_classes
    P1 = layers.Conv2D(
        num_classes, 1, activation='softmax', name='final_conv'
    )(P1)
    
    model = models.Model(inputs=inputs, outputs=P1)
    
    return model

    