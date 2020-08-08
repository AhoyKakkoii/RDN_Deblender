import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import warnings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tqdm import tqdm
from skimage.io import imread
from skimage.util import img_as_float
from skimage.transform import AffineTransform, rescale, warp, rotate

from deblender.utils import np_to_tfrecords


plt.rcParams['figure.figsize'] = [8, 8]
warnings.filterwarnings('ignore')


def crop(x, h=80, w=80):
    assert x.shape[0] >= h, x.shape[1] >= w
    ch, cw = int((x.shape[0]-h)/2), int((x.shape[1]-w)/2)
    crop_img = x[ch:x.shape[0]-ch, cw:x.shape[1]-cw, :]
#    print('after crop:', crop_img, crop_img.shape, type(crop_img))
    return x[ch:x.shape[0]-ch, cw:x.shape[1]-cw, :]


def downsample(x, factor=3.):
#    down_sample = rescale(x, (1, 1, 1./factor), mode='constant')
#    print ('after downsample', down_sample, down_sample.shape, type(down_sample))
#    return rescale(x, 1./factor, mode='constant')
    return rescale(x, (1, 1, 1), mode='constant')

def crop_and_downsample(x):
    return downsample(crop(x))


def perturb(x):
    sx, sy = np.array(x.shape[:2])//2

    rotation = np.random.uniform(0, 2*np.pi)
    scale = np.power(np.e, np.random.uniform(-1, 0.5))
    v_flip = np.random.choice([True, False])
    h_flip = np.random.choice([True, False])
    shift = np.concatenate([np.arange(-50, -10), np.arange(10, 50)])
    translation = np.random.choice(shift, size=2)

    if v_flip:
        x = x[::-1, :, :]
    if h_flip:
        x = x[:, ::-1, :]

    shift = AffineTransform(translation=[-sx, -sy])
    inv_shift = AffineTransform(translation=[sx, sy])

    tform = AffineTransform(
        scale=[scale, scale],
        rotation=rotation,
        translation=translation
    )

    return warp(x, (shift + tform + inv_shift).inverse)


def merge(x1, x2):
    assert np.shape(x1) == np.shape(x2)
    y = [np.max(np.dstack([x1[:, :, i], x2[:, :, i]]), -1)
        for i in range(x1.shape[-1])]
    return np.dstack(y)


def get_batch(batch_size, training=True):
    if training:
        filepath_individual = "../galaxy_zoo/individuals_2blend_train"
        filepath_merged = "../galaxy_zoo/merged_2blend_train"
    else:
        filepath_individual = "../galaxy_zoo/individuals_2blend_valid"
        filepath_merged = "../galaxy_zoo/merged_2blend_valid"


    print('Generating data from: {} and {}'.format(filepath_individual, filepath_merged))
    files = os.listdir(filepath_merged)

    x_1or2 = np.random.randint(1, high = 3, size = len(files))
    y_1or2 = [int((-1)*x_12+3) for x_12 in x_1or2]
    x_files = ['single' + file[5: 13] + '_' + str(x_1or2[i]) + '.png' for i, file in enumerate(files)]
    y_files = ['single' + file[5: 13] + '_' + str(y_1or2[i]) + '.png' for i, file in enumerate(files)]

    x_images = [img_as_float(imread(os.path.join(filepath_individual, file))) for file in x_files]
    y_images = [img_as_float(imread(os.path.join(filepath_individual, file))) for file in y_files]

    print(x_images[0].shape, type(x_images[0]), len(x_images))
    '''
    following is to find the sample that has a higher mean luminance
    '''
    brighter = []
    darker = []
    for image1, image2 in zip(x_images, y_images):

        mean1 = image1.mean()
        mean2 = image2.mean()

        if mean1 >= mean2:
            brighter.append(image1)
            darker.append(image2)
        else:
            darker.append(image1)
            brighter.append(image2)

    x = [crop_and_downsample(image) for image in brighter]
    y = [crop_and_downsample(perturb(image)) for image in darker]
#    y = [crop_and_downsample(image) for image in y_images]

    merged = [merge(x1, x2) for x1, x2 in zip(x, y)]

    x = 2*np.array(x) - 1
    y = 2*np.array(y) - 1
    merged = np.array(merged)

    x = x.reshape(batch_size, -1)
    y = y.reshape(batch_size, -1)
    merged = merged.reshape(batch_size, -1)

    return x, y, merged


def plot_batch(batch):
    batch_size = len(batch[0])
    columns = len(batch)

    fig, axes = plt.subplots(
        columns,
        batch_size,
        figsize=(2*batch_size, 2*columns)
    )

    for i in range(columns):
        for j in range(batch_size):
            ax = axes[i][j]
            ax.imshow(batch[i][j])
            ax.set_aspect('equal')
            ax.axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()

def main():
    tfrecords_path = './data/train'

    if not os.path.isdir(tfrecords_path):
        os.makedirs(tfrecords_path)

    for i in tqdm(range(1)):

        x, y, z = get_batch(8000, training=True)
        filename = os.path.join(tfrecords_path, 'train-batch_{}'.format(i+1))
        print(x.dtype)
        data_dict = {'blended': z, 'x': x, 'y': y}
        np_to_tfrecords(data_dict, filename)

    tfrecords_path = './data/valid'

    if not os.path.isdir(tfrecords_path):
        os.makedirs(tfrecords_path)

    for i in tqdm(range(1)):

        x, y, z = get_batch(2000, training=False)
        filename = os.path.join(tfrecords_path, 'valid-batch_{}'.format(i+1))
        data_dict = {'blended': z, 'x': x, 'y': y}
        np_to_tfrecords(data_dict, filename)


if __name__ == '__main__':
    main()
