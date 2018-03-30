import numpy as np
import keras
import os
from PIL import Image, ImageOps
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage import affine_transform

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, path, batch_size=4, dim=(256,256), n_channels=1, shuffle=True,
                 rotation=False, flipping=False, mirror_edges=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.path = path
        self.rotation = rotation
        self.flipping = flipping
        self.mirror_edges = mirror_edges
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.mirror_edges:
            out_shape = (self.dim[0] + self.mirror_edges, self.dim[0] + self.mirror_edges)

        X = np.empty((self.batch_size, *out_shape, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_channels))

        if self.rotation:
            rot = np.random.choice([0, 90, 180, 270], self.batch_size)
        else:
            rot = np.zeros((self.batch_size))

        if self.flipping:
            flip = np.random.choice([True, False], (2,self.batch_size))
        else:
            flip = np.zeros((2, self.batch_size), dtype=bool)


        # Generate data
        for i, sample in enumerate(list_IDs_temp):

            with Image.open(os.path.join(self.path, sample, 'images', '{}.png'.format(sample))) as x_img:
                x_img = x_img.resize(self.dim)
                x_img = x_img.rotate(rot[i])
                if flip[0,i]:
                    x_img = x_img.transpose(Image.FLIP_LEFT_RIGHT)
                if flip[1,i]:
                    x_img = x_img.transpose(Image.FLIP_TOP_BOTTOM)
                x_img = x_img.convert(mode='L')
                x_img = ImageOps.autocontrast(x_img)
                x_arr = np.array(x_img) / 255
                if self.mirror_edges:
                    x_arr = affine_transform(x_arr, [1,1], offset=[self.mirror_edges/2, self.mirror_edges/2],
                                             output_shape=out_shape, mode='mirror')
                x_arr = np.expand_dims(x_arr, axis=2)
                X[i,] = x_arr

            with Image.open(os.path.join(self.path, sample, 'mask', '{}.png'.format(sample))) as y_img:
                y_img = y_img.resize(self.dim)
                y_img = y_img.rotate(rot[i])
                if flip[0,i]:
                    y_img = y_img.transpose(Image.FLIP_LEFT_RIGHT)
                if flip[1,i]:
                    y_img = y_img.transpose(Image.FLIP_TOP_BOTTOM)
                y_arr = np.array(y_img) / 255
                y_arr = np.expand_dims(y_arr, axis=2)
                Y[i,] = y_arr

        return X, Y

if __name__ == '__main__':
    training = os.listdir('img')[0:8]
    training_generator = DataGenerator(training, 'img')

