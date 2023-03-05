import cv2
from queue import Queue
from lueysloops.net.search.duckduckgo_image_search import get_image_list
from lueysloops.net.search.duckduckgo_image_search import GetImagesFromImageSearchReturn
import numpy as np
import os
import numba

CHANNEL_INDEX_IN_SHAPE = 2

@numba.jit()
def decompose_fft_image(image, bins_to_keep=10):
    image_mod = np.copy(image)

    for channel_index in range(0, image_mod.shape[CHANNEL_INDEX_IN_SHAPE]):
        # Take FFT of channel mat
        image_fft = np.fft.fft2(image_mod[:,:,channel_index])
        for row_index in range(0, image_fft.shape[0]):
            sorted_vector = np.argsort(image_fft[row_index][:])
            image_fft[row_index][sorted_vector[bins_to_keep::]] = 0 + 0j
        image_mod[:,:,channel_index] = np.fft.ifft2(image_fft).real

    return image_mod


def main():
    image_queue = Queue()
    GetImagesFromImageSearchReturn(get_image_list("doggie", 1000), image_queue)

    index = 0
    path = "out/"
    os.makedirs(path, exist_ok=True)
    while image_queue.qsize() != 0:
        current_image = image_queue.get()
        a = decompose_fft_image(current_image, bins_to_keep=900)
        cv2.imwrite(path + str(index) + "_mod.png", np.abs(a))
        cv2.imwrite(path + str(index) + "_org.png", current_image)
        index += 1

if __name__ == "__main__":
    main()
