import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image as image
import random

# Collect and preprocess data


def preprocess_image(image):
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image


celeb_img = image.load_img('/home/kaan/Elif Hoca Assignments/AubreyPlaza.jpg')
style_img = image.load_img('/home/kaan/Elif Hoca Assignments/brick.jpg')
celeb_img = preprocess_image(celeb_img)
target_img = preprocess_image(celeb_img)
style_img_resized = tf.image.resize(style_img, [256, 256])

# Define model architecture


def create_model():
    # Define input layer
    input_layer = tf.keras.layers.Input(shape=(256, 256, 3))

    # Content model
    content_model = tf.keras.applications.VGG19(
        include_top=False, weights='imagenet')
    content_layers = content_model.get_layer('block4_conv2').output
    content_model = tf.keras.models.Model(
        inputs=content_model.input, outputs=content_layers)
    content_model.trainable = False
    content_features = content_model(input_layer)

    # Style model
    style_model = tf.keras.applications.VGG19(
    include_top=False, weights='imagenet', input_tensor=input_layer)
    style_layers = [
        style_model.get_layer('block1_conv1').output,
        style_model.get_layer('block2_conv1').output,
        style_model.get_layer('block3_conv1').output,
        style_model.get_layer('block4_conv1').output,
        style_model.get_layer('block5_conv1').output,
    ]
    style_model = tf.keras.models.Model(
        inputs=input_layer, outputs=style_layers)
    style_model.trainable = False


    # Resize style images
    style_features_resized = [tf.image.resize(
        style_layer, [256, 256]) for style_layer in style_layers]

    # Adjust number of feature maps in content model to match style model
    num_filters = style_layers[-1].shape[-1]
    content_features = tf.keras.layers.Conv2D(
        num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(content_features)

    # Add instance normalization layer
    output = tf.keras.layers.Lambda(lambda x: (
        x - tf.reduce_mean(x, axis=[1, 2], keepdims=True)) / tf.math.reduce_std(x, axis=[1, 2], keepdims=True))(content_features)

    # Resize output to 256x256
    output = tf.keras.layers.Lambda(
        lambda x: tf.image.resize(x, [256, 256]))(output)

    # Output model
    output_layers = tf.keras.layers.Concatenate()(
        style_features_resized + [output])
    output_model = tf.keras.models.Model(
        inputs=input_layer, outputs=output_layers)

    return output_model


model = create_model()

# Define loss function


def content_loss(output, target):
    return tf.reduce_mean(tf.square(output - target))


def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def style_loss(output, target):
    output_grams = [gram_matrix(out) for out in tf.split(output, num_or_size_splits=5, axis=3)]
    target_grams = [gram_matrix(out) for out in tf.split(target, num_or_size_splits=5, axis=3)]
    style_loss_per_layer = [tf.reduce_mean(tf.square(output_grams[i] - target_grams[i])) for i in range(5)]
    style_loss_val = tf.reduce_mean(style_loss_per_layer)
    return style_loss_val

