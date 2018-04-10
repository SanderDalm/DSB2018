import tensorflow as tf
import numpy as np


def IOU(x, y):
    sum_array = np.round(x + y)
    intersection = len(sum_array[sum_array == 2])
    union = intersection + len(sum_array[sum_array == 1])

    if union > 0:
        return intersection / union
    else:
        return 0


class NeuralNet(object):

    def __init__(self, height, width, channels, batchgen):

        self.batchgen = batchgen

        self.graph = tf.Graph()

        self.session = tf.Session()  # config=tf.ConfigProto(log_device_placement=True)

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channels], name='input')
        self.x = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.x)

        self.dropout_rate = tf.placeholder(tf.float32)

        self.prediction = self.UNET(self.x, self.dropout_rate)

        self.label = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])
        self.boundaries = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])
        #self.label = tf.subtract(self.label, tf.minimum(self.boundaries, 1))

        self.loss = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=self.label,
                                    logits=self.prediction),
                                    tf.add(self.boundaries, 1)))

        #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #                            labels=self.label,
        #                           logits=self.prediction))


        self.prediction = tf.nn.sigmoid(self.prediction)
        self.lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)
        self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)
        self.saver = tf.train.Saver(max_to_keep=None,
                                    name='checkpoint_saver')

    def Conv2D(self, x, filters, kernel_size, stride, padding='same'):
        return tf.layers.conv2d(inputs=x,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=stride,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.keras.initializers.he_normal(),
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                activity_regularizer=tf.keras.regularizers.l2(l=0.01),
                                padding=padding)



    def UpConv2D(self, x, filters, kernel_size, stride, padding='same'):
        return tf.layers.conv2d_transpose(inputs=x,
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          strides=stride,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                          activity_regularizer=tf.keras.regularizers.l2(l=0.01),
                                          padding=padding)




    def UNET(self, x, dropout_rate):

        # Convolutional layers
        filter_size = 16
        conv1 = self.Conv2D(x, filter_size, 3, 1)
        conv1 = self.Conv2D(conv1, filter_size, 3, 1)

        pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

        conv2 = self.Conv2D(pool1, filter_size * 2, 3, 1)
        conv2 = self.Conv2D(conv2, filter_size * 2, 3, 1)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

        conv3 = self.Conv2D(pool2, filter_size * 4, 3, 1)
        conv3 = self.Conv2D(conv3, filter_size * 4, 3, 1)
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2)

        conv4 = self.Conv2D(pool3, filter_size * 8, 3, 1)
        conv4 = self.Conv2D(conv4, filter_size * 8, 3, 1)
        dropout4 = tf.layers.dropout(conv4, rate=dropout_rate)
        pool4 = tf.layers.max_pooling2d(dropout4, pool_size=2, strides=2)

        conv5 = self.Conv2D(pool4, filter_size * 16, 3, 1)
        conv5 = self.Conv2D(conv5, filter_size * 16, 3, 1)
        dropout5 = tf.layers.dropout(conv5, rate=dropout_rate)

        # Upconvolutional layers
        upconv6 = self.UpConv2D(dropout5, filter_size * 8, 2, 2)
        concat6 = tf.concat([dropout4, upconv6], axis=3)
        conv6 = self.Conv2D(concat6, filter_size * 8, 3, 1)
        conv6 = self.Conv2D(conv6, filter_size * 8, 3, 1)

        upconv7 = self.UpConv2D(conv6, filter_size * 4, 2, 2)
        concat7 = tf.concat([conv3, upconv7], axis=3)
        conv7 = self.Conv2D(concat7, filter_size * 4, 3, 1)
        conv7 = self.Conv2D(conv7, filter_size * 4, 3, 1)

        upconv8 = self.UpConv2D(conv7, filter_size * 2, 2, 2)
        concat8 = tf.concat([conv2, upconv8], axis=3)
        conv8 = self.Conv2D(concat8, filter_size * 2, 3, 1)
        conv8 = self.Conv2D(conv8, filter_size * 2, 3, 1)

        upconv9 = self.UpConv2D(conv8, filter_size, 2, 2)
        concat9 = tf.concat([conv1, upconv9], axis=3)
        conv9 = self.Conv2D(concat9, filter_size, 3, 1)
        conv9 = self.Conv2D(conv9, filter_size, 3, 1)
        conv9 = self.Conv2D(conv9, 2, 3, 1)

        return tf.layers.conv2d(inputs=conv9,
                                filters=1,
                                kernel_size=1,
                                strides=1,
                                activation=None,  # tf.nn.sigmoid,
                                kernel_initializer=tf.keras.initializers.he_normal(),
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                activity_regularizer=tf.keras.regularizers.l2(l=0.01),
                                padding='same')



    def train(self, num_steps, batch_size, dropout_rate, lr, decay, checkpoint='models/neural_net'):

        loss_list = []
        val_loss_list = []
        val_iou_list = []

        for step in range(num_steps):

            x_batch, y_batch, boundary_batch = self.batchgen.generate_batch(batch_size)

            feed_dict = {
                         self.x: x_batch,
                         self.label: y_batch,
                         self.boundaries : boundary_batch,
                         self.dropout_rate: dropout_rate,
                         self.lr: lr
                         }

            loss_, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
            lr *= decay

            if step % 100 == 0:

                x_batch, y_batch, boundaries_batch = self.batchgen.generate_val_data()
                feed_dict = {
                             self.x: x_batch,
                             self.label: y_batch,
                             self.boundaries: boundaries_batch,
                             self.dropout_rate: 0
                             }
                val_loss = self.session.run([self.loss], feed_dict=feed_dict)
                val_loss_list.append(val_loss)
                loss_list.append(loss_)
                print('step: {}'.format(step))
                print('train loss: {}'.format(loss_))
                print('val loss: {}'.format(val_loss))
                print('lr: {}'.format(lr))
                print('')
                val_iou_list.append(self.validate())

            if (step+1) % 1000 == 0 or step == num_steps-1:
                self.saver.save(self.session, checkpoint + str(step) + '.ckpt')
                print('Saved to {}'.format(checkpoint + str(step) + '.ckpt'))

        return loss_list, val_loss_list, val_iou_list


    def load_weights(self, checkpoint):

        self.saver.restore(self.session, checkpoint)


    def predict(self, x):

        return self.session.run([self.prediction], {self.x: x,
                                                    self.dropout_rate: 0
                                                    })[0]


    def validate(self):

        x_val, y_val, _ = self.batchgen.generate_val_data()
        preds = self.predict(x_val)

        IOU_list = []
        for index, pred in enumerate(preds):
            IOU_score = IOU(pred, y_val[index])
            IOU_list.append(IOU_score)
        IOU_array = np.array(IOU_list)

        mean_iou = np.mean(IOU_array)
        print('Mean val IOU: {}'.format(mean_iou))

        return mean_iou