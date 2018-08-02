import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from skimage import io, transform, color

class GooseDataset():
    def __init__(self):
        self.n_data = 1000
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.img_dir = os.path.join(self.path, 'images')
        self.ann_dir = os.path.join(self.path, 'annotations')
        self.cropped_img_dir = os.path.join(self.path, 'cropped_images')
        self.processed_img_dir = os.path.join(self.path, 'processed_images')
        self.default_shape = [533, 800]
        self.default_cropped_shape = [105, 195]

    def crop_images(self):
        img_files = [os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))]
        ann_files = [os.path.join(self.ann_dir, f) for f in os.listdir(self.ann_dir) if os.path.isfile(os.path.join(self.ann_dir, f))]
        if os.path.isdir(self.cropped_img_dir):
            shutil.rmtree(self.cropped_img_dir)
        os.mkdir(self.cropped_img_dir)

        margin = 10
        for i, file in enumerate(ann_files):
            # Open image
            img = io.imread(img_files[i])
        
            # Read XML
            in_file = open(file)
            tree = ET.parse(in_file)
            root = tree.getroot() 
            imsize = root.find('size')
            w = int(imsize.find('width').text)
            h = int(imsize.find('height').text)
            
            j = 0
            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                xn = int(float(xmlbox.find('xmin').text))
                xx = int(float(xmlbox.find('xmax').text))
                yn = int(float(xmlbox.find('ymin').text))
                yx = int(float(xmlbox.find('ymax').text))
                
                xn = max(0, xn - margin)
                xx = min(w, xx + margin)
                yn = max(0, yn - margin)
                yx = min(h, yx + margin)
                
                cropped_img = img[yn:yx,xn:xx]
                cropped_img_file = os.path.join(self.cropped_img_dir, img_files[i][len(self.img_dir) + 1:])
                if (j > 0):
                    cropped_img_file = cropped_img_file[:-4] + '-%d'.format(j) + cropped_img_file[-4:]
                io.imsave(cropped_img_file, cropped_img)

                j = j + 1

    def read_image(self, dir, img_no, grayscale=False, resize_shape=None, save=False):
        file = os.path.join(dir, 'goose-mugshot-{0:04d}.jpg'.format(img_no))
        img = io.imread(file)
        shape = img.shape
        if resize_shape:
            img = transform.resize(img, resize_shape)
            shape = img.shape
        if grayscale:
            img = color.rgb2gray(img)
            n_channels = 1
        else:
            n_channels = 3
        if save:
            out_file = os.path.join(self.processed_img_dir, 'goose-mugshot-{0:04d}.jpg'.format(img_no))
            io.imsave(out_file, img)

        img = img.reshape([shape[0], shape[1], n_channels])
        return img

    def load_data(self, test_ratio=0.2, grayscale=False, cropped=False, resize_shape=None, save=False):
        if grayscale:
            n_channels = 1
        else:
            n_channels = 3

        if cropped:
            dir = self.cropped_img_dir
            if not os.path.isdir(self.cropped_img_dir):
                self.crop_images()
            if not resize_shape:
                resize_shape = self.default_cropped_shape
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

        x_train = np.zeros([self.n_data - n_test, shape[0], shape[1], n_channels])
        y_train = np.zeros([self.n_data - n_test, 1]).astype(int)
        x_test = np.zeros([n_test, shape[0], shape[1], n_channels])
        y_test = np.zeros([n_test, 1]).astype(int)

        if not grayscale:
            x_train = x_train.astype(int)
            x_test = x_test.astype(int)

        for i in range(len(arr_test)):
            img = self.read_image(dir=dir, img_no=arr_test[i], grayscale=grayscale, resize_shape=resize_shape, save=save)
            x_test[i,:,:,:] = img
            y_test[i] = 1
            
        for i in range(len(arr_train)):
            img = self.read_image(dir=dir, img_no=arr_train[i], grayscale=grayscale, resize_shape=resize_shape, save=save)
            x_train[i,:,:,:] = img
            y_train[i] = 1

        return (x_train, y_train), (x_test, y_test)