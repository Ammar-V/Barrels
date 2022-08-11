from skimage.color import rgb2ycbcr
import cv2
import numpy as np


def post_processing(image):
    img = image.astype('uint8')
    new = cv2.bilateralFilter(img, -1, 7, 7)
    new = cv2.blur(img, (7, 7))
    new = (new > 0.5).astype('uint8')

    return new


def mask_barrel(image, delta_a=0, delta_b=0, labelling=True, name='') -> np.ndarray:
    """
    This function takes an input image and ouputs a segmentation mask for red barrels.
    The mask consists of an intersection between mask_a and mask_b. mask_a is computed \
        by taking the combining the Cr, S, Y, and V channels in a particular manner.\
        Thresholding is performed on this to produce a binary mask. mask_b uses a similar techinque\
        but instead utilizes the RGB channel. The B and G channesl are combined (mean)\
        and then compared with the red channel. The thresholding works as follows:\
        if R - BG > some value, then set the pixel to 1 (i.e. what is the redness of this pixesl).\
        This operation separates red pixesl from white pixels. The masks are then combined and returned.

    """

    # Split into YCbCr channels
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_a = rgb2ycbcr(img)
    y, cb, cr = cv2.split(img_a)

    # Split into HSV channels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    # return v
    s *= 255  # Change s from 0 - 1 to 0 - 255

    # Combine different channels before thresholding for mask_a
    mask_a_pre = cr - s - y
    mask_a_pre = np.clip(mask_a_pre, 0, 255)

    # Peform thresholding for mask_a
    brightness = v.mean()
    min, max = delta_a, delta_a + 20
    _, mask_a = cv2.threshold(
        mask_a_pre, min, max, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    mask_a = mask_a.astype("float32")

    # Gives a measure of how bright the red patches are
    brightness_mask = (np.clip(mask_a, 0, 1) * v).mean()

    # A white pixel is: (255, 255, 255) whereas a red pixel is (255, 0, 0)
    # Filter by all the pixels that are pure red (255, 0, 0)
    r, g, b = cv2.split(img)
    mask_b = np.zeros(r.shape)
    bg = np.mean(np.array([b, g]), axis=0)

    # Threshold to get mask_b
    if brightness_mask > 20:  # larger number means red color is spread out
        thresh_b = int(brightness*0.5) + 30

    elif brightness_mask > 3:  # range for barrel in normal lighting conditions
        thresh_b = int(brightness * 0.5)

    else:  # If barrel is insanely tiny
        thresh_b = int(brightness*0.5) - 30

    thresh_b += delta_b

    # Bound it between suitable values
    thresh_b = np.clip(thresh_b, 0, 255)

    # Perform thresholding for mask_b
    mask_b = ((r - bg) > thresh_b).astype('float')

    # Intersection of the two masks gives best segmentation
    mask = mask_a * mask_b

    # Show masks for the labelling program
    if labelling:
        cv2.imshow("mask_a", mask_a)
        cv2.imshow("mask_b", mask_b)

    mask = np.clip(mask, 0, 1)

    print(f"{name=}, {brightness=}, {min=}, {max=}, {brightness_mask=}, {thresh_b=}")

    return post_processing(mask)
