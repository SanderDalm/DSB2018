import os
import sys
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb
from skimage.segmentation import find_boundaries


class BatchGenerator(object):

    def __init__(self, height, width, channels, data_dir_train, data_dir_test, submission_run):

        self.height = height
        self.width = width
        self.channels = channels

        self.x_train, self.y_train, self.boundaries_train = self.read_train_data(data_dir_train)
        #self.shuffle()
        self.x_val, self.y_val, self.boundaries_val = self.x_train[:100], self.y_train[:100], self.boundaries_train[:100]


        # In we're not in submission mode we won't train on the validation data
        if not submission_run:
           self.x_train, self.y_train, self.boundaries_train = self.x_train[100:], self.y_train[100:], self.boundaries_train[100:]

        self.x_test = self.read_test_data(data_dir_test)

        self.cursor = 0


    def read_train_data(self, data_dir):

        train_ids = next(os.walk(data_dir))[1]
        images = np.zeros((len(train_ids), self.height, self.width, self.channels), dtype=np.float64)
        labels = np.zeros((len(train_ids), self.height, self.width, 1), dtype=np.bool)
        boundaries = np.zeros((len(train_ids), self.height, self.width, 1), dtype=np.uint8)
        sys.stdout.flush()

        print('Getting and resizing train images ... ')
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

            path = data_dir + id_
            img, _ = self.read_image(path + '/images/' + id_ + '.png')
            images[n] = img
            mask = np.zeros((self.height, self.width, 1), dtype=np.bool)
            mask_boundaries = np.zeros((self.height, self.width, 1), dtype=np.uint8)

            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (self.height, self.width), mode='constant',
                                              preserve_range=True), axis=-1)
                mask_ = (mask_/255).astype(np.uint8)

                # Add mask
                mask = np.maximum(mask, mask_)

                # Add boundry
                mask_boundaries_ = find_boundaries(mask_, mode='outer').astype(np.uint8)
                mask_boundaries = np.add(mask_boundaries, mask_boundaries_)

            labels[n] = mask
            boundaries[n] = mask_boundaries

        x_train = images
        y_train = labels

        return x_train, y_train, boundaries.astype(np.float32)


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
        boundary_batch = []

        for _ in range(batch_size):

            x = self.x_train[self.cursor]
            y = self.y_train[self.cursor]
            boundary = self.boundaries_train[self.cursor]
            self.cursor += 1
            if self.cursor == len(self.x_train):
                self.shuffle()
                self.cursor = 0

            x, y, boundary = self.augment(x, y, boundary)
            x_batch.append(x)
            y_batch.append(y)
            boundary_batch.append(boundary)

        return np.array(x_batch).reshape([batch_size,
                       self.height,
                       self.width,
                       self.channels]), np.array(y_batch).reshape([batch_size,self.height,self.width,1]), np.array(boundary_batch).reshape([batch_size,self.height,self.width,1])


    def generate_val_data(self):

        return self.x_val, self.y_val, self.boundaries_val


    def generate_test_data(self):

        return self.x_test


    def augment(self, x, y, boundary):

        mirror = np.random.rand()

        if mirror > .5:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)
            boundary = np.flip(boundary, axis=1)

        rotation = np.random.randint(0,4)
        x = np.rot90(x, rotation)
        y = np.rot90(y, rotation)
        boundary = np.rot90(boundary, rotation)

        #x += np.random.normal(0, .01, [x.shape[0], x.shape[1], 1])

        return x, y, boundary