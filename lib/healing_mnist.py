"""
Data loader for the Healing MNIST data set (c.f. https://arxiv.org/abs/1511.05121)

Adapted from https://github.com/Nikita6000/deep_kalman_filter_for_BM/blob/master/healing_mnist.py
"""


import numpy as np
import scipy.ndimage
from tensorflow.keras.datasets import mnist


def apply_square(img, square_size):
    img = np.array(img)
    img[:square_size, :square_size] = 255
    return img


def apply_noise(img, bit_flip_ratio):
    img = np.array(img)
    mask = np.random.random(size=(28,28)) < bit_flip_ratio
    img[mask] = 255 - img[mask]
    return img


def get_rotations(img, rotation_steps):
    for rot in rotation_steps:
        img = scipy.ndimage.rotate(img, rot, reshape=False)
        yield img


def binarize(img):
    return (img > 127).astype(np.int)


def heal_image(img, seq_len, square_count, square_size, noise_ratio, max_angle):
    squares_begin = np.random.randint(0, seq_len - square_count)
    squares_end = squares_begin + square_count

    rotations = []
    rotation_steps = np.random.normal(size=seq_len, scale=max_angle)

    for idx, rotation in enumerate(get_rotations(img, rotation_steps)):
        # Don't add the squares right now
        # if idx >= squares_begin and idx < squares_end:
        #     rotation = apply_square(rotation, square_size)
        
        # Don't add noise for now
        # noisy_img = apply_noise(rotation, noise_ratio)
        noisy_img = rotation
        binarized_img = binarize(noisy_img)
        rotations.append(binarized_img)

    return rotations, rotation_steps


class HealingMNIST():
    def __init__(self, seq_len=5, square_count=3, square_size=5, noise_ratio=0.15, digits=range(10), max_angle=180):
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        mnist_train = [(img,label) for img, label in zip(x_train, y_train) if label in digits]
        mnist_test = [(img, label) for img, label in zip(x_test, y_test) if label in digits]

        train_images = []
        test_images = []
        train_rotations = []
        test_rotations = []
        train_labels = []
        test_labels = []

        for img, label in mnist_train:
            train_img, train_rot = heal_image(img, seq_len, square_count, square_size, noise_ratio, max_angle)
            train_images.append(train_img)
            train_rotations.append(train_rot)
            train_labels.append(label)

        for img, label in mnist_test:
            test_img, test_rot = heal_image(img, seq_len, square_count, square_size, noise_ratio, max_angle)
            test_images.append(test_img)
            test_rotations.append(test_rot)
            test_labels.append(label)
        
        self.train_images = np.array(train_images)
        self.test_images = np.array(test_images)
        self.train_rotations = np.array(train_rotations)
        self.test_rotations = np.array(test_rotations)
        self.train_labels = np.array(train_labels)
        self.test_labels = np.array(test_labels)