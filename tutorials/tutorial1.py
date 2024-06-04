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

desired_aruco_dictionary = "DICT_6X6_250"
 
# The different ArUco dictionaries built into the OpenCV library. 
ARUCO_DICT = {
  "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def display_vector_field(
    filename: Union[pathlib.Path, str],
    on_img: Optional[bool] = False,
    image_name: Optional[Union[pathlib.Path, str]] = None,
    window_size: Optional[int] = 32,
    scaling_factor: Optional[float] = 1.,
    ax: Optional[Any] = None,
    width: Optional[float] = 0.0025,
    show_invalid: Optional[bool] = True,
    save_name=None,
    **kw
):
    """ Displays quiver plot of the data stored in the file 


    Parameters
    ----------
    filename :  string
        the absolute path of the text file

    on_img : Bool, optional
        if True, display the vector field on top of the image provided by 
        image_name

    image_name : string, optional
        path to the image to plot the vector field onto when on_img is True

    window_size : int, optional
        when on_img is True, provide the interrogation window size to fit the 
        background image to the vector field

    scaling_factor : float, optional
        when on_img is True, provide the scaling factor to scale the background
        image to the vector field

    show_invalid: bool, show or not the invalid vectors, default is True


    Key arguments   : (additional parameters, optional)
        *scale*: [None | float]
        *width*: [None | float]


    See also:
    ---------
    matplotlib.pyplot.quiver


    Examples
    --------
    --- only vector field
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt',scale=100, 
                                           width=0.0025) 

    --- vector field on top of image
    >>> openpiv.tools.display_vector_field(Path('./exp1_0000.txt'), on_img=True, 
                                          image_name=Path('exp1_001_a.bmp'), 
                                          window_size=32, scaling_factor=70, 
                                          scale=100, width=0.0025)

    """
    # print(f' Loading {filename} which exists {filename.exists()}')
    a = np.loadtxt(filename)
    # parse
    x, y, u, v, flags, mask = a[:, 0], a[:,
                                         1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if on_img is True:  # plot a background image
        im = tools.imread(image_name)
        im = tools.negative(im)  # plot negative of the image for more clarity
        xmax = np.amax(x) + window_size / (2 * scaling_factor)
        ymax = np.amax(y) + window_size / (2 * scaling_factor)
        ax.imshow(im, cmap="Greys_r", extent=[0.0, xmax, 0.0, ymax])

    # first mask whatever has to be masked
    u[mask.astype(bool)] = 0.
    v[mask.astype(bool)] = 0.

    # now mark the valid/invalid vectors
    invalid = flags > 0  # mask.astype("bool")
    valid = ~invalid

    # visual conversion for the data on image
    # to be consistent with the image coordinate system

    # if on_img:
    #     y = y.max() - y
    #     v *= -1

    ax.quiver(
        x[valid],
        y[valid],
        u[valid],
        v[valid],
        color="b",
        width=width,
        **kw
    )

    if show_invalid and len(invalid) > 0:
        ax.quiver(
            x[invalid],
            y[invalid],
            u[invalid],
            v[invalid],
            color="r",
            width=width,
            **kw,
        )

    # if on_img is False:
    #     ax.invert_yaxis()

    ax.set_aspect(1.)
    # fig.canvas.set_window_title('Vector field, '+str(np.count_nonzero(invalid))+' wrong vectors')
    # plt.show()
    plt.savefig(save_name)
    print("comp")
    return fig, ax

def process_single(image1, image2, num):
    # we can run it from any folder
    path = files('openpiv') / "data" / "test1"/"Pre_Processed" 
    print(path)
    path2=files('openpiv') / "data" / "test1" / "Output"
    #The images can be read using the imread function, and diplayed with matplotlib.
    frame_a  = tools.imread( path / image1 )
    frame_b  = tools.imread( path / image2 )

    frame_a = frame_a.astype(np.int32)
    frame_b = frame_b.astype(np.int32)
    #PIV cross correlation algorithm, allow overlap when searching using interrogation window, make search area bigger
    #return u and v vector component, the higher sig2noise then higher probability u and v are correct
    u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
        window_size=32, overlap=16, dt=1, search_area_size=64, sig2noise_method='peak2peak' )

    print(u,v,sig2noise)
    #get coordinate get the center of each interrogation window
    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, search_area_size=64, overlap=16 )
    #Vectors below a certain threshold are substituted by NaN, bascially masking
    flags_s2n = validation.sig2noise_val(sig2noise, threshold = 1.2 )

    flags_g = validation.global_val( u, v, (-10, 10), (-10, 10) )
    flags = flags_s2n | flags_g
    #find outlier vectors, and substitute them by an average of neighboring vectors. 
    #The larger the kernel_size the larger is the considered neighborhood
    u, v = filters.replace_outliers( u, v, flags, method='localmean', max_iter=30, kernel_size=2)
    #convert pixels to millimeters
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
    # 0,0 shall be bottom left, positive rotation rate is counterclockwise
    x, y, u, v = tools.transform_coordinates(x, y, u, v)
    #save the vector field to a ASCII tabular file
    tools.save(str(path2 / f'data_{num}.vec') , x, y, u, v, flags)
    # tools.display_vector_field(path / 'test_data.vec', scale=75, width=0.0035)
    print("test")
    #plots and saves the velocity plot for current image
    display_vector_field(
        str(path2 / f'data_{num}.vec'), 
        scale=1, 
        scaling_factor=96.52,
        width=0.0035,
        on_img=True,
        image_name = str(path / image1),
        save_name=str(path2/f'processed_{num}')
    )
    print("test2")
# # path of video file
# video_path = r'data/test1/vid_2000-07-27_03-16-59.mp4'

# # Open video file
# video = cv2.VideoCapture(video_path)

# # number of frames in video
# #frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# frame_count = 200
# # Convert frame to image and save to file
# for i in range(frame_count):
#     ret, frame = video.read()
#     if ret and (i%10 ==0):
#         image_path = f"data/test1/Video_To_Frame/image_{i}.jpg"
#         cv2.imwrite(image_path, frame)

# # Close video file
# video.release()
# print("finished")
# import the necessary packages



  
def ArucoMarker(image_path):
    """
    Main method of the program.
    """
    # Check that we have a valid ArUco marker
    if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
        # print("[INFO] ArUCo tag of '{}' is not supported".format(
        #     args["type"]))
        sys.exit(0)
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str,
        default="DICT_ARUCO_ORIGINAL",
        help="type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    # Load the ArUco dictionary
    # print("[INFO] detecting '{}' markers...".format(
    #     desired_aruco_dictionary))
    this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[desired_aruco_dictionary])
    this_aruco_parameters = cv2.aruco.DetectorParameters_create()
     
    # Read the image
    frame = cv2.imread(image_path)
    
    # Get the frame size
    frame_size = (frame.shape[1], frame.shape[0])
    
    # Detect ArUco markers in the image
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        frame, this_aruco_dictionary, parameters=this_aruco_parameters)
    
    # print(rejected)
    # print("pp")
    # Check that at least one ArUco marker was detected
    if len(corners) > 0:
        # print("identified")
        # Flatten the ArUco IDs list
        ids = ids.flatten()
         
        # Create an empty array to store the marker locations
        marker_locations = []
         
        # Loop over the detected ArUco corners
        for (marker_corner, marker_id) in zip(corners, ids):
         
            # Extract the marker corners
            corners = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners
             
            # Convert the (x,y) coordinate pairs to integers
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
             
            # Draw the bounding box of the ArUco detection
            cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)
             
            # Calculate and draw the center of the ArUco marker
            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
             
            # Draw the ArUco marker ID on the image
            # The ID is always located at the top_left of the ArUco marker
            cv2.putText(frame, str(marker_id), 
                (top_left[0], top_left[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            # print("entered")
            # print(center_x)
            # print(center_y)
            
            # Append the marker location to the array
            marker_locations.append((center_x, center_y))
    # Return the array of marker locations and frame size
    return marker_locations, frame_size

def crop_image(image_path, adjusted_frame_width, adjusted_frame_height, distance_above_marker, distance_left_of_marker):
    """
    Main method of the program.
    """
    # Check that we have a valid ArUco marker
    if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
        sys.exit(0)

    # Load the ArUco dictionary
    this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[desired_aruco_dictionary])
    this_aruco_parameters = cv2.aruco.DetectorParameters_create()
     
    # Read the image
    frame = cv2.imread(image_path)
    
    # Get the frame size
    frame_size = (frame.shape[1], frame.shape[0])
    
    # Detect ArUco markers in the image
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        frame, this_aruco_dictionary, parameters=this_aruco_parameters)
    
    # Check that at least one ArUco marker was detected
    if len(corners) > 0:
        # Flatten the ArUco IDs list
        ids = ids.flatten()
         
        # Create an empty array to store the marker locations
        marker_locations = []
         
        # Loop over the detected ArUco corners
        for (marker_corner, marker_id) in zip(corners, ids):
         
            # Extract the marker corners
            corners = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners
             
            # Convert the (x,y) coordinate pairs to integers
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
             
            # Calculate and draw the center of the ArUco marker
            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            
            # Append the marker location to the array
            marker_locations.append((center_x, center_y))
    print(marker_locations)
    # Calculate the crop coordinates
    left_x = marker_locations[0][0] - distance_left_of_marker
    left_y = marker_locations[0][1] - distance_above_marker
    right_x = left_x + adjusted_frame_width
    right_y = left_y + adjusted_frame_height
    
    # Crop the image
    cropped_image = frame[left_y:right_y, left_x:right_x]
    
    return cropped_image
    
def process_deviation ():
    dir_path = r'data/test1/Test'
    list_image = os.listdir(dir_path)
    path = files('openpiv') / "data" / "test1"/"Video_To_Frame" 
    # print(list_image)
    #pre-processing
    left_marker_x = []
    left_marker_y = []
    right_marker_x = []
    right_marker_y = []
    frame_sizes = []

    for i,_ in enumerate(os.listdir(dir_path)):
        marker_locations, frame_sizes = ArucoMarker(os.path.join(path,list_image[i]))
        left_marker_x.append(marker_locations[0][0])
        left_marker_y.append(marker_locations[0][1])
        right_marker_x.append(marker_locations[1][0])
        right_marker_y.append(marker_locations[1][1])

        # Calculate the maximum deviation between the markers
        max_dev = (max(left_marker_x) - min(left_marker_x), max(right_marker_y) - min(right_marker_y))

    # Calculate the adjusted allowed frame size
    adjusted_frame_width = max(frame_sizes) - max_dev[0]
    adjusted_frame_height = min(frame_sizes) - max_dev[1]

    min_in_Y = min(left_marker_y)
    min_in_X = min(left_marker_x)

    distance_above_marker = min(frame_sizes) -min_in_Y
    distance_left_of_marker = max(frame_sizes) - min_in_X
    print(min_in_Y,min_in_X)
    for i,_ in enumerate(os.listdir(dir_path)):
        cropped_image = crop_image(os.path.join(path,list_image[i]), adjusted_frame_width, adjusted_frame_height, min_in_Y, min_in_X)
        output_path = f'data/test1/Cropped/{i}.png'
        cv2.imwrite(output_path, cropped_image)

def particle_masking():
    path = files('openpiv') / "data" / "test1"/"Cropped" 
    list_image = os.listdir(path)
    print(list_image)
    #pre-processing
    for i,_ in enumerate(os.listdir(path)):
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

        # print(list_image[i])
        # print(nemo)
        # print("test")
        # processed = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
        processed = cv2.cvtColor(nemo, cv2.COLOR_BGR2YCrCb)
        # (Y,Cr,Cb)=cv2.split(processed)
        hsv_processed= cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
        (H,S,V)=cv2.split(hsv_processed)
        # cv2.imshow("Red",H)
        # # cv2.imshow("Sat",S)
        # # cv2.imshow("Val",V)
        kernel = np.ones((2,2),np.uint8)
        mod_img = cv2.adaptiveThreshold(H,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,6)
        modded= cv2.morphologyEx(mod_img, cv2.MORPH_CLOSE, kernel)
        output_path = f'data/test1/Pre_Processed/{i}.png'
        cv2.imwrite(output_path, modded)

process_deviation()
particle_masking()

input_path = r'data/test1/Pre_Processed'
input_list_image = os.listdir(input_path)
#openpiv
(marker_locations, frameSize)= ArucoMarker(input_list_image[0])
left_marker_x= marker_locations[0][0]
left_marker_y=marker_locations[0][1]
right_marker_x=marker_locations[1][0]
right_marker_y=marker_locations[1][1]
distance_in_pixels = left_marker_x-right_marker_x
distance_in_mm = 17.988
for i,_ in enumerate(os.listdir(input_path)[:-1]):
    process_single(input_list_image[i], input_list_image[i+1],i)
print("fini")
