import os
os.chdir('/home/sander/datascience/DSB2018/DSB2018')


import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import label, opening, remove_small_holes
from skimage.transform import resize
import pandas as pd

from tensorflow_implementation.neural_net import NeuralNet
from tensorflow_implementation.batch_generator import BatchGenerator


###########################
# Training
###########################

SIZE = 256

batchgen = BatchGenerator(height=SIZE,
                          width=SIZE,
                          channels=1,
                          data_dir_train='stage1_train/',
                          data_dir_test='stage1_test/',
                          submission_run=False)

x_train, y_train, boundaries_train = batchgen.x_train, batchgen.y_train, batchgen.boundaries_train
x_val = batchgen.x_val
x_test, test_ids, sizes_test = batchgen.x_test


#x,y,b=batchgen.generate_batch(32)
#plt.imshow(x[15].reshape(SIZE, SIZE), cmap='gray')

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

model = NeuralNet(SIZE, SIZE, 1, batchgen)
loss_list, val_loss_list, val_iou_list = model.train(num_steps=16000,
             batch_size=16,
             dropout_rate=0,
             lr=.001,
             decay=.9998,
             checkpoint='/home/sander/datascience/DSB2018/DSB2018/tensorflow_implementation/models/')

# Load weights
model.load_weights('/home/sander/datascience/DSB2018/DSB2018/tensorflow_implementation/models/final.ckpt')

plt.plot(loss_list)
plt.plot(val_loss_list)
plt.plot(val_iou_list)
plt.legend(['Train loss', 'Val loss', 'Val IOU'])
plt.show()

###########################
# Validatie
###########################

x_val, y_val, b_val = batchgen.generate_val_data()
val_preds = model.predict(x_val)
index = 50

test=val_preds[index]
test=test.flatten()
plt.hist(test)

plt.imshow(x_val[index].reshape(SIZE, SIZE), cmap='gray')
plt.imshow(y_val[index].reshape(SIZE, SIZE), cmap='gray')
plt.imshow(b_val[index].reshape(SIZE, SIZE), cmap='gray')
plt.imshow(val_preds[index].reshape(SIZE, SIZE), cmap='gray')
plt.imshow(np.round(val_preds[index].reshape(SIZE, SIZE)), cmap='gray')


def IOU(x, y):

    sum_array = np.round(x+y)
    intersection = len(sum_array[sum_array == 2])
    union = intersection + len(sum_array[sum_array == 1])
    if union > 0:
        return intersection/union
    else:
        return 0

IOU_list = []
for index, pred in enumerate(val_preds):
    IOU_score = IOU(pred, y_val[index])
    print(index, IOU_score)
    IOU_list.append(IOU_score)
IOU_array = np.array(IOU_list)

print(np.mean(IOU_array))

################
# 8x voorspellen
################

def get_preds(x):
    def get_rotated_predictions(img, flip_back=False):

        preds = []
        for rotation in [0, 1, 2, 3]:

            rotated_img = np.rot90(img, rotation)
            pred_rotated = model.predict(rotated_img.reshape(1, SIZE, SIZE, 1))
            pred_rotated_reversed = np.rot90(pred_rotated.reshape(SIZE, SIZE), -rotation)
            if flip_back:
                pred_rotated_reversed = np.flip(pred_rotated_reversed, axis=1)
            pred_rotated_reversed = pred_rotated_reversed.reshape(SIZE, SIZE, 1)
            preds.append(pred_rotated_reversed)
        return preds

    preds = []

    for sample in x:

        preds_for_sample = []

        img = sample.reshape(SIZE, SIZE)
        preds_for_sample.extend(get_rotated_predictions(img))

        img_mirr = np.flip(img, axis=1)
        preds_for_sample.extend(get_rotated_predictions(img_mirr, flip_back=True))

        total_pred_for_sample = np.concatenate(preds_for_sample, axis=2)
        pred = np.mean(total_pred_for_sample, axis=2)
        preds.append(pred)
    return preds

    #plt.imshow(x_test[0].reshape(SIZE, SIZE), cmap='gray')
    #plt.imshow(preds[0].reshape(SIZE, SIZE), cmap='gray')

preds = get_preds(x_test)


###########################
# Submission
###########################

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

preds = [opening(x) for x in preds]
preds = [remove_small_holes(np.round(x).astype(np.uint8)) for x in preds]

def rle_encoding(x):

    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


preds_test_upsampled = []
for i in range(len(preds)):
    preds_test_upsampled.append(resize(np.squeeze(preds[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

new_test_ids = []
rles = []

for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub.csv', index=False)


#index = 1
#plt.imshow(resize(np.squeeze(preds[index]), (sizes_test[index][0], sizes_test[index][1]), mode='constant', preserve_range=True))
#plt.imshow(preds_test_upsampled[index], cmap='gray')