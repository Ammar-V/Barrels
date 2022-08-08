from mask_classical import mask_barrel
import cv2

import os

INPUT_DIR = 'images/'
OUTPUT_DIR = 'output/'


def create_masks(input_dir, output_dir):

    for file in os.listdir(input_dir):
        if file.endswith('.png'):  # Skip sub folders

            img = cv2.imread(os.path.join(input_dir, file))

            mask = mask_barrel(img, labelling=False, name=file) * 255  # Mask
            cv2.imwrite(os.path.join(output_dir, file), mask)  # Save mask


if __name__ == '__main__':

    # Make output dir if it doesn't exist
    try:
        os.mkdir(OUTPUT_DIR)
    except:
        pass
    create_masks(INPUT_DIR, OUTPUT_DIR)
