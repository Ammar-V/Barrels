from torch import outer
from mask_classical import mask_barrel
import cv2

import os

INPUT_DIR = 'images/'
OUTPUT_DIR = 'masks/'


def create_masks(input_dir, output_dir):

    for file in os.listdir(input_dir):
        if file.endswith('.png'):

            img = cv2.imread(os.path.join(input_dir, file))
            # h, w, _ = img.shape
            # img = cv2.resize(img, (256, 256))

            mask = mask_barrel(img, labelling=False, name=file) * 255
            # mask = cv2.resize(mask, (w, h), cv2.INTER_AREA)
            cv2.imwrite(os.path.join(output_dir, f'mask_{file}'), mask)


if __name__ == '__main__':

    try:
        os.mkdir(OUTPUT_DIR)
    except:
        pass
    create_masks(INPUT_DIR, OUTPUT_DIR)
