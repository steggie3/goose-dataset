import os
import shutil
import numpy as np
from skimage import io, transform, color

class GooseDataset():
    def __init__(self):
        self.n_data = 1000
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.img_dir = os.path.join(self.path, 'images')
        self.cropped_img_dir = os.path.join(self.path, 'cropped_images')
        self.processed_img_dir = os.path.join(self.path, 'processed_images')
        self.default_shape = [800, 533]

    def read_image(self, dir, img_no, grayscale=False, resize_shape=None, save=False):
        file = os.path.join(dir, 'goose-mugshot-{0:04d}.jpg'.format(img_no))
        img = io.imread(file)
        shape = img.shape
        if resize_shape:
            img = transform.resize(img, resize_shape)
            shape = img.shape
        if grayscale:
            img = color.rgb2gray(img)
        if save:
            out_file = os.path.join(self.processed_img_dir, 'goose-mugshot-{0:04d}.jpg'.format(img_no))
            io.imsave(out_file, img)
        if grayscale:
            img = img.reshape([shape[0], shape[1], 1])
        return img

    def load_data(self, test_ratio=0.2, grayscale=False, cropped=False, resize_shape=None, save=False):
        if grayscale:
            n_channels = 1
        else:
            n_channels = 3

        if cropped:
            dir = self.cropped_img_dir
        else:
            dir = self.img_dir

        if resize_shape:
            shape = resize_shape
        else:
            shape = self.default_shape

        if save:
            if os.path.isdir(self.processed_img_dir):
                shutil.rmtree(self.processed_img_dir)
            os.mkdir(self.processed_img_dir)

        n_test = int(self.n_data * test_ratio)
        arr = np.random.permutation(self.n_data) + 1
        arr_test = np.sort(arr[0:n_test])
        arr_train = np.sort(arr[n_test:self.n_data])
        #print(arr_test.shape)
        #print(arr_train.shape)

        x_train = np.zeros([self.n_data - n_test, shape[0], shape[1], n_channels])
        y_train = np.zeros([self.n_data - n_test, 1])
        x_test = np.zeros([n_test, shape[0], shape[1], n_channels])
        y_test = np.zeros([n_test, 1])

        for i in range(len(arr_test)):
            img = self.read_image(dir=dir, img_no=arr_test[i], grayscale=grayscale, resize_shape=resize_shape, save=save)
            x_test[i,:,:,:] = img
            y_test[i] = 1
            
        for i in range(len(arr_train)):
            img = self.read_image(dir=dir, img_no=arr_train[i], grayscale=grayscale, resize_shape=resize_shape, save=save)
            x_train[i,:,:,:] = img
            y_train[i] = 1
        
        #print(x_train.shape)

        return (x_train, y_train), (x_test, y_test)