import os
os.chdir('/home/sander/datascience/DSB2018/DSB2018')

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import label
from skimage.transform import resize
import pandas as pd

from tensorflow_implementation.neural_net import NeuralNet
from tensorflow_implementation.batch_generator import BatchGenerator


###########################
# Training
###########################

SIZE = 256
MIRROR = 0

batchgen = BatchGenerator(height=SIZE,
                          width=SIZE,
                          channels=1,
                          data_dir_train='stage1_train/',
                          data_dir_test='stage1_test/',
                          mirror_edges=MIRROR,
                          submission_run=False)

x_train, y_train = batchgen.x_train, batchgen.y_train
x_val = batchgen.x_val
x_test, test_ids, sizes_test = batchgen.x_test


#x = x_train[0].reshape([SIZE+MIRROR,SIZE+MIRROR])
#plt.imshow(x, cmap='gray')
#plt.imshow(x, cmap='gray')
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

model = NeuralNet(SIZE, SIZE, 1, MIRROR, batchgen)

#model.load_weights('/home/sander/kaggle/models/neural_net2500.ckpt')

loss_list, val_loss_list, val_iou_list = model.train(num_steps=2000,
             batch_size=64,
             dropout_rate=0,
             lr=.0001,
             decay=1,
             checkpoint='models/neural_net')

plt.plot(loss_list)
plt.plot(val_loss_list)
plt.plot(val_iou_list)
plt.legend(['Train loss', 'Val loss', 'Val IOU'])
plt.show()

###########################
# Validatie
###########################

x_val, y_val = batchgen.generate_val_data()
val_preds = model.predict(x_val)
index = 1

plt.imshow(x_val[index].reshape(SIZE, SIZE), cmap='gray')
plt.imshow(y_val[index].reshape(SIZE, SIZE), cmap='gray')
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

###########################
# Submission
###########################

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

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

preds = model.predict(x_test)
preds = np.round(preds)
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


index = 1
plt.imshow(resize(np.squeeze(preds[index]), (sizes_test[index][0], sizes_test[index][1]), mode='constant', preserve_range=True))
plt.imshow(preds_test_upsampled[index], cmap='gray')
