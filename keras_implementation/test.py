from skimage import measure, draw
from skimage.segmentation import mark_boundaries, find_boundaries


import matplotlib.pyplot as plt

from PIL import Image

import numpy as np
import os

path = 'C:/Users/huubh/Documents/DSB2018_bak/img/'

aa = Image.open(os.path.join(path, '94519eb45cbe1573252623b7ea06a8b43c19c930f5c9b685edb639d0db719ab0/masks/13ad7b7af69b33743a71f302f6220cac7ed8cbf165051b6269910071a763e9cb.png'))
_aa = aa.resize((256,256))
_aa = np.array(_aa)
ll = measure.find_contours(_aa, 1)

print(ll)


one_sample = os.path.join(path,'94519eb45cbe1573252623b7ea06a8b43c19c930f5c9b685edb639d0db719ab0','masks')

all_masks = os.listdir(one_sample)

full_bounds = np.zeros((256,256))

for i, mask in enumerate(all_masks):
    empty = np.zeros((256,256))
    with Image.open(os.path.join(one_sample, mask)) as _img:
        _array = np.array(_img.resize((256, 256)))
        _array = np.array(_array > 0, dtype=int) * (i + 1)
        full_bounds = np.add(full_bounds, _array)

full_bounds = np.array(full_bounds, dtype=int)

ll = find_boundaries(label_img = full_bounds, connectivity = 1, mode='outer', background=0)

ziezo = np.array(ll, dtype=int)
ziezo = ziezo *  255

mask_image = Image.fromarray(ziezo.astype('uint8'), 'L')
mask_image.save('test.png')





# complete_mask = np.maximum(complete_mask, _mask)