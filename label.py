from mask_classical import mask_barrel

from skimage.color import rgb2ycbcr
import numpy as np
import cv2
import os

INPUT_DIR = "./images"

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

            mask = mask_barrel(img, delta_a, delta_b)

            while True:
                cv2.imshow("Original", img)

                # If the selection is empty keep the original mask.
                # Only update the mask once the selection is complete, ie selection_start is False
                # if np.any(selection) and not selection_start:
                #     selection_mask = selection * mask
                # else:
                selection_mask = mask

                # selection_mask = np.clip(selection_mask, 0, 1)
                cv2.imshow("mask", selection_mask * 255)
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
