import os
import numpy as np
from PIL import Image

def find_all_samples(path):
    all_samples = os.listdir(path)
    return all_samples

def check(path, width, height):
    samples = find_all_samples(path)
    for sample in samples:
        sample_path = os.path.join(path, sample)
        sample_path_masks = os.path.join(sample_path, 'masks')
        masks = os.listdir(sample_path_masks)
        complete_mask = np.zeros((width, height), dtype=float)
        for mask in masks:
            with Image.open(os.path.join(sample_path_masks, mask)) as _mask:
                _mask = _mask.resize((width, height))
                _mask = np.array(_mask) /255
                complete_mask = np.add(complete_mask, _mask)
                # print(np.max(complete_mask))
        if np.max(complete_mask) > 1:
            print('help')
        # os.mkdir(os.path.join(sample_path, 'mask'))
        # mask_image = Image.fromarray(complete_mask.astype('uint8'), 'L')
        # mask_image.save(os.path.join(sample_path, 'mask', '{}.png'.format(sample)))

check('C:/Users/huubh/Documents/DSB2018_bak/img', 256, 256)