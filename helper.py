# Importing libraries
import cv2

# Helper functions
def create_gaussian_pyramid(img, num_levels):
    # Initializing the list
    pyramid = [img]

    for i in range(num_levels):
        # reduce operation
        img = cv2.pyrDown(img)
        pyramid.append(img)

    return pyramid

def reconstruction(pyramid, idx, num_levels, height, width):
    filtered_img = pyramid[idx]

    for i in range(num_levels):
        # upscaling filtered images
        filtered_img = cv2.pyrUp(filtered_img)

    filtered_img = filtered_img[:height, :width]
    return filtered_img