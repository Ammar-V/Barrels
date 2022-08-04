from pydoc import Doc
from cv2 import threshold
from skimage.color import rgb2ycbcr
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os

INPUT_DIR = "./images"


def mask_barrel(image, delta_a=0, delta_b=0, labelling=True, name='') -> np.ndarray:
    """
    This function takes an input image and ouputs a segmentation mask for red barrels.
    The mask consists of an intersection between mask_a and mask_b. mask_a is computed \
        by taking the difference between the Cr and Y channels. mask_b uses a similar techinque\
        but instead utilizes the RGB channel. The B and G channesl are combined (mean)\
        and then compared with the red channel. The thresholding works as follows:\
        if R - BG > some value, then set the pixel to 1. This condition separates \
        red pixesl from white pixels. The masks are then combined and returned.

    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_a = rgb2ycbcr(img)
    y, _, cr = cv2.split(img_a)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    s *= 255

    cr_y = cr - s - y
    cr_y *= v
    cr_y = np.clip(cr_y, 0, 255)

    # Peform thresholding on cr_y for mask_a
    brightness = v.mean()
    min, max = delta_a, delta_a + 20
    _, mask_a = cv2.threshold(
        cr_y, min, max, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    mask_a = mask_a.astype("float32")

    # A white pixel is: (255, 255, 255) whereas a red pixel is (255, 0, 0)
    # Filter by all the pixels that are pure red (255, 0, 0)

    calculation = (np.clip(mask_a, 0, 1) * v).mean()
    # cv2.imshow('tester', (np.clip(mask_a, 0, 1) * v).astype('uint8'))

    r, g, b = cv2.split(img)
    mask_b = np.zeros(r.shape)
    bg = np.mean(np.array([b, g]), axis=0)

    # Threshold to get mask_b
    if calculation > 20:  # larger number means red color is spread out
        thresh_b = int(brightness*0.5) + 30

    elif calculation > 3:  # range for barrel in normal lighting conditions
        thresh_b = int(brightness * 0.5)

    else:  # If barrel is insanely tiny
        thresh_b = int(brightness*0.5) - 30

    thresh_b += delta_b

    # Bound it between suitable values
    thresh_b = np.clip(thresh_b, 0, 255)

    mask_b = ((r - bg) > thresh_b).astype('float')

    print(f"{name=}, {brightness=}, {thresh_b=}, {calculation=}, {np.max(mask_a)=}")

    # Intersection of the two masks
    mask = mask_a * mask_b

    if labelling:
        cv2.imshow("mask_a", mask_a)
        cv2.imshow("mask_b", mask_b)

    mask = np.clip(mask, 0, 1)

    print(f"{name=}, {brightness=}, {min=}, {max=}, {thresh_b=}")

    return mask


# Set global variales for mask selection
selection_start, selection = False, None

# Event handler to detect the selections


def select_area(event, x, y, flags, params):
    global selection_start
    global selection

    # print(f"Left click at {x} and {y}")
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if selection_start:
            selection_start = False
        else:
            selection_start = x, y
        print(f"Left click at {x} and {y}")
    else:
        if selection_start:
            selection = cv2.rectangle(
                selection, selection_start, [x, y], 1, -1)


def run_mask_app(size=256):
    global selection
    # Make a folder to store the masks
    try:
        os.mkdir(f"{INPUT_DIR}/masks")
    except:
        pass

    for image in os.listdir(INPUT_DIR):
        if os.path.isfile(os.path.join(INPUT_DIR, image)):

            mask_name = "mask_" + image

            img = cv2.imread(f"{INPUT_DIR}/{image}")
            img = cv2.resize(img, (size, size))

            # Reset for new image
            selection = np.zeros((size, size))
            delta_a, delta_b = 0, 0
            img_a = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_a = rgb2ycbcr(img_a)
            y, _, cr = cv2.split(img_a)

            mask = mask_barrel(img, delta_a, delta_b)

            while True:
                cv2.imshow("test", img)

                # If the selection is empty keep the original mask.
                # Only update the mask once the selection is complete, ie selection_start is False
                if np.any(selection) and not selection_start:
                    selection_mask = selection * mask
                else:
                    selection_mask = mask

                selection_mask = np.clip(selection_mask, 0, 1)
                cv2.imshow("mask", selection_mask)
                cv2.setMouseCallback("mask", select_area)

                # Window controls
                k = cv2.waitKey(1)

                if k == 27:  # end process
                    cv2.destroyAllWindows()
                    quit()
                elif k == 101:  # e saves mask and goes to next image
                    cv2.imwrite(f"{INPUT_DIR}/masks/{mask_name}",
                                selection_mask * 255)
                    break
                elif k == 119:  # w increments thresh_a
                    delta_a += 5
                    mask = mask_barrel(img, delta_a, delta_b)

                elif k == 115:  # s decrements thresh_a
                    delta_a -= 5
                    mask = mask_barrel(img, delta_a, delta_b)

                elif k == 97:  # a decrements thresh_b
                    delta_b -= 5
                    mask = mask_barrel(img, delta_a, delta_b)

                elif k == 100:  # d arrow decrements thresh_b
                    delta_b += 5
                    mask = mask_barrel(img, delta_a, delta_b)

                elif k == 114:  # Reset selection
                    selection = np.zeros((size, size))
                elif k == 32:  # spacebar to go to next image
                    break
                else:
                    pass


if __name__ == "__main__":
    run_mask_app()
