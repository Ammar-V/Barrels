from cv2 import threshold
from skimage.color import rgb2ycbcr
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os

INPUT_DIR = "./images"


def mask_barrel(image, inc_a=0, inc_b=0) -> np.ndarray:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_a = rgb2ycbcr(img)

    # Convert into YCbCr channels
    y, _, cr = cv2.split(img_a)

    # Subtract the y channel from the cr
    cr_y = np.array(cr) - np.array(y)
    cr_y = np.clip(cr_y, 0, 255)

    # Thresholding for mask_a

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    brightness = v.mean()

    thresh_a = brightness // 3
    thresh_a += inc_a
    min, max = brightness - thresh_a, brightness + thresh_a

    _, mask_a = cv2.threshold(cr_y, min, max, cv2.THRESH_BINARY)
    mask_a = mask_a.astype("float32")

    # A white pixel is: (255, 255, 255) whereas a red pixel is (255, 0, 0)
    # Filter by all the pixels that are pure red
    r, g, b = cv2.split(img)
    mask_b = np.zeros(r.shape)
    bg = np.mean(np.array([b, g]), axis=0)

    thresh_b = int(130 * brightness / 255)
    thresh_b += inc_b
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if r[i][j] - bg[i][j] > thresh_b:
                mask_b[i][j] = 1

    # Intersection of the two masks
    mask = mask_a * mask_b
    cv2.imshow("mask_a", mask_a)
    cv2.imshow("mask_b", mask_b)
    cv2.imshow("mask", mask)

    print(f"{thresh_a=}, {min=}, {max=}, {thresh_b=}")

    return mask


if __name__ == "__main__":

    # Make a folder to store the masks
    try:
        os.mkdir(f"{INPUT_DIR}/masks")
    except:
        pass

    for image in os.listdir(INPUT_DIR):
        if os.path.isfile(os.path.join(INPUT_DIR, image)):

            mask_name = "mask_" + image

            img = cv2.imread(f"{INPUT_DIR}/{image}")
            img = cv2.resize(img, (256, 256))

            delta_a, delta_b = 0, 0
            while True:
                cv2.imshow("test", img)
                mask = mask_barrel(img, delta_a, delta_b)

                # Window controls
                k = cv2.waitKey(0)

                if k == 27:  # end process
                    cv2.destroyAllWindows()
                    quit()
                elif k == 32:  # spacebar saves mask and goes to next image
                    cv2.imwrite(f"{INPUT_DIR}/masks/{mask_name}", mask)
                    break
                elif k == 119:  # w increments thresh_a
                    print("this key")
                    delta_a += 5
                elif k == 115:  # s decrements thresh_a
                    delta_a -= 5
                elif k == 97:  # a decrements thresh_b
                    delta_b -= 5
                elif k == 100:  # d arrow decrements thresh_b
                    delta_b += 5
                else:
                    pass
