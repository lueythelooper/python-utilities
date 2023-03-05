import requests
from duckduckgo_search import ddg_images
import cv2
import numpy as np
from threading import Thread
from queue import Queue
import sys

import argparse

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def GetImagesFromImageSearchReturn(image_search_return_objects, queue_image):
    for obj in image_search_return_objects:
        # Currently downloading whole image
        img_link = obj['image']
        # Currently download
        # img_link = obj['thumbnail']
        #print (obj['image'])
        try:
            img_data = requests.get(img_link, verify=False, timeout=4).content
            mat = cv2.imdecode(np.asarray(bytearray(img_data), dtype=np.uint8), 1)
            if mat is not None:
                queue_image.put(mat)

        except Exception as e:
            print (str(e))

    return

def get_image_list(search_term : str, max_results: int):
    return ddg_images(keywords=search_term, safesearch='Off', region='wt-wt', max_results=max_results)


def main():    
    # Initialize parser
    parser = argparse.ArgumentParser(description = "Duckduckgo image viewer")

    # Optional argument for num threads
    parser.add_argument("-t", "--threads", help = "Number of image download threads")
    parser.add_argument("-s", "--search", help = "The image search string", required=True)
    parser.add_argument("-n", "--image_num", help = "Number of images to attempt download", required=True)

    args = parser.parse_args()

    threads = 5
    if args.threads is not None:
        threads = args.threads
    print ("Number of threads: ", threads)


    image_list = get_image_list(args.search, int(args.image_num))
    queue_of_images = Queue()

    IMAGES_PER_THREAD = int(len(image_list) / 5)

    thread_list = []

    done = False
    index = 0
    while (done == False):
        if len(image_list) - index > IMAGES_PER_THREAD:
            thread_list.append(Thread(target=GetImagesFromImageSearchReturn, args=(image_list[index:index+IMAGES_PER_THREAD], queue_of_images, )))
            index += IMAGES_PER_THREAD
        else:
            thread_list.append(Thread(target=GetImagesFromImageSearchReturn, args=(image_list[index:len(image_list)], queue_of_images, )))
            done = True

    for thread_obj in thread_list:
        thread_obj.start()

    for thread_obj in thread_list:
        thread_obj.join()

    ur = []
    while(queue_of_images.empty() == False):
        ur.append(queue_of_images.get())

    index = 0
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while (True):
        if ur[index].shape[0] > 1000: 
            cv2.imshow('frame', ResizeWithAspectRatio(image=ur[index], height=1000))
        else:
            cv2.imshow('frame', ur[index])

        doogie = cv2.waitKey(1) & 0xFF
        if doogie == ord('q'):
            print ("Got quit break")
            break
        if doogie == ord('k'):
            print ("Previous image")
            index += 1
            if index > len(ur) - 1:
                index = 0
            print (index)
        if doogie == ord('j'):
            print ("Next image")
            index -= 1
            if index < 0:
                index = len(ur)-1
            print (index)
        if doogie == ord('b'):
            index = 0

    print ("Exiting")

if __name__ == "__main__":
    main()
