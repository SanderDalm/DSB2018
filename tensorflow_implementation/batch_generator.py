import os
import sys
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb
from scipy.ndimage import affine_transform


class BatchGenerator(object):

    def __init__(self, height, width, channels, data_dir_train, data_dir_test, submission_run,
                 mirror_edges=False):

        self.height = height
        self.width = width
        self.channels = channels
        self.mirror_edges = mirror_edges

        self.x_train, self.y_train = self.read_train_data(data_dir_train)
        self.shuffle()
        self.x_val, self.y_val = self.x_train[:100], self.y_train[:100]


        # In we're not in submission mode we won't train on the validation data
        if not submission_run:
           self.x_train, self.y_train = self.x_train[100:], self.y_train[100:]

        self.x_test = self.read_test_data(data_dir_test)

        self.cursor = 0


    def read_train_data(self, data_dir):

        train_ids = next(os.walk(data_dir))[1]
        images = np.zeros((len(train_ids), self.height+self.mirror_edges, self.width+self.mirror_edges, self.channels), dtype=np.float64)
        labels = np.zeros((len(train_ids), self.height, self.width, 1), dtype=np.bool)
        sys.stdout.flush()

        print('Getting and resizing train images ... ')
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

            path = data_dir + id_
            img, _ = self.read_image(path + '/images/' + id_ + '.png')
            if self.mirror_edges > 0:
                img = self.perform_mirror_edges(img)
            images[n] = img
            mask = np.zeros((self.height, self.width, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (self.height, self.width), mode='constant',
                                              preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
            labels[n] = mask

        x_train = images
        y_train = labels

        return x_train, y_train.astype(np.float64)


    def read_test_data(self, data_dir):

        test_ids = next(os.walk(data_dir))[1]
        x_test = np.zeros((len(test_ids), self.height, self.width, self.channels), dtype=np.float64)
        sizes_test = []
        print('Getting and resizing test images ... ')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            path = data_dir + id_
            img, original_size = self.read_image(path + '/images/' + id_ + '.png')
            sizes_test.append(original_size)
            x_test[n] = img

        return x_test, test_ids, sizes_test


    def read_image(self, path):

        img = imread(path)
        try:
            img = rgb2gray(rgba2rgb(img))
        except:
            img = rgb2gray(img)

        original_size = (img.shape[0], img.shape[1])
        img = resize(img, (self.height, self.width), mode='constant', preserve_range=True)

        return img.reshape([self.height, self.width, self.channels]), original_size


    def shuffle(self):

        indices = np.random.permutation(len(self.x_train))
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]
        #print("Every day I'm shuffling.")


    def generate_batch(self, batch_size):


        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            x = self.x_train[self.cursor]
            y = self.y_train[self.cursor]
            self.cursor += 1
            if self.cursor == len(self.x_train):
                self.shuffle()
                self.cursor = 0

            x, y = self.augment(x, y)
            x_batch.append(x)
            y_batch.append(y)

        return np.array(x_batch).reshape([batch_size,
                       self.height+self.mirror_edges,
                       self.width+self.mirror_edges,
                       self.channels]), np.array(y_batch).reshape([batch_size,self.height,self.width,1])


    def generate_val_data(self):

        return self.x_val, self.y_val


    def generate_test_data(self):

        return self.x_test


    def augment(self, x, y):

        flip_hor = np.random.rand()
        flip_ver = np.random.rand()
        if flip_hor > .5:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)
        if flip_ver > .5:
            x = np.flip(x, axis=0)
            y = np.flip(y, axis=0)

        return x, y


    def perform_mirror_edges(self, img):

        img = img.reshape([self.width, self.height])
        img = affine_transform(img,
                 [1,1],
                 output_shape=[self.width+self.mirror_edges, self.height+self.mirror_edges],
                 mode='mirror')
        img = img.reshape([self.width+self.mirror_edges, self.height+self.mirror_edges, self.channels])

        return img