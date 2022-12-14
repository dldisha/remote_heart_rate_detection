#Importing libraries
import sys
import numpy as np
import cv2

#Helper functions
def create_gaussian_pyramid(frame, levels):
    #Initializing the list
    pyramid = [frame]

    for level in range(levels):
        #reduce operation
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)

    return pyramid

def reconstruction(pyramid, index, levels, video_height, video_width):
    filtered_frame = pyramid[index]

    for level in range(levels):
        #upscaling filtered frames
        filtered_frame = cv2.pyrUp(filtered_frame)

    filtered_frame = filtered_frame[:video_height, :video_width]
    return filtered_frame