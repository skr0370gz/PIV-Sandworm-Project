from importlib_resources import files
import numpy as np
from openpiv import tools, pyprocess, scaling, validation, filters
import cv2 

import sys
import pathlib
import multiprocessing
from typing import Any, Union, List, Optional
# import re

import matplotlib.pyplot as plt
import matplotlib.patches as pt
from natsort import natsorted

# from builtins import range
from imageio.v3 import imread as _imread, imwrite as _imsave
import os

import argparse
import imutils

dir_path = r'data/test1/Video_To_Frame'
list_image = os.listdir(dir_path)
path = files('openpiv') / "data" / "test1"/"Video_To_Frame" 
print(list_image)
#pre-processing
for i,_ in enumerate(os.listdir(dir_path)):
    #process_single(list_image[i], list_image[i+1],i)
    #img = cv.imread(list_image[i], cv.IMREAD_GRAYSCALE)
    print(path / list_image[i])
    nemo = cv2.imread(os.path.join(path,list_image[i]))
    # store image shape(h, w) = img.shape


    scale_percent = 50 # percent of original sizewidth = int(img.shape[1] * scale_percent / 100) 

    height = int(nemo.shape[0] * scale_percent / 100) 
    width = int(nemo.shape[1] * scale_percent / 100) 

    dim = (width, height) 


    nemo = cv2.resize(nemo, dim, interpolation = cv2.INTER_AREA) 


    
    print(list_image[i])
    print(nemo)
    print("test")
    processed = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
    hsv_processed= cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
    (H,S,V)=cv2.split(hsv_processed)
    # cv2.imshow("HUE",H)
    # cv2.imshow("Sat",S)
    # cv2.imshow("Val",V)
    mod_img = cv2.adaptiveThreshold(S,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,4)
    cv2.imshow("image", mod_img)
    cv2.imshow("nemo", nemo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # light_blue = (213.0769, 98.1132, 62.3529)
    # dark_blue = (211.4706, 100.0000, 80.0000)
    # light_yellow = (130, 63, 185)  # Adjust these values based on your specific blue
    # dark_yellow = (179, 255, 255)
    # mask = cv2.inRange(hsv_processed, light_yellow, dark_yellow)
    # result = cv2.bitwise_and(processed, processed, mask=mask)
    # plt.imshow(mask)
    # plt.savefig(f'data/test1/Output/{i}')
    output_path = f'data/test1/Pre_Processed/{i}.png'
    plt.imsave(output_path, hsv_processed)