#Importing libraries
import sys
import numpy as np
import cv2

#import python files modules
import helper_fn

#Show Webcam 
camera = None
camera = cv2.VideoCapture(0) #read the video

#Webcam parameters   
total_width = 320
total_height = 240
video_width = 160
video_height = 120
channels = 3
frame_rate = 15

#Setting video frame
camera.set(6, total_width)
camera.set(8, total_height)
camera.set(10, video_width)
camera.set(8, video_height)

#Color Magnification Parameters
levels = 3
alpha = 170
min_frequency = 1.0
max_frequency = 2.0
buffer_size = 150
buffer_index = 0

#Heart rate text display Parameters
font_style = cv2.FONT_HERSHEY_DUPLEX
font_scale = 1
font_color = (253, 3, 3)
init_text_location = (160, 35)
HR_text_location = (video_width//2 + 160, 35)
box_color = (5, 5, 5)
box_weight = 4

#Heart Rate calculation variables
bpm_frequency = 10
bpm_buffer_index = 2
bpm_buffer_size = 10
bpm_buffer = np.zeros((bpm_buffer_size))

#Saving the Output Heart Rate video
output_file = "output.mov"
output_writer = cv2.VideoWriter()
output_writer.open(output_file, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), frame_rate, (total_width, total_height), True)

#Initialize Gaussian Pyramid
first_frame = np.zeros((video_height, video_width, channels))
first_gaussian_pyramid = helper_fn.create_gaussian_pyramid(first_frame, levels+1)[levels]
gaussian_video = np.zeros((buffer_size, first_gaussian_pyramid.shape[0], first_gaussian_pyramid.shape[1], channels))
#Fourier Transform Average
fourier_transform_avg = np.zeros((buffer_size))

#Bandpass filter
frequencies = (1.0*frame_rate) * np.arange(buffer_size) / (1.0*buffer_size)
mask = (frequencies >= min_frequency) & (frequencies <= max_frequency)

#ALGO
i = 0

while (True):
    init, frame = camera.read()
    #frame = cv2.resize(frame, (600, 400))
    if init == False:
        break

    detection_frame = frame[video_height//2:total_height - video_height//2, video_width//2:total_width - video_width//2, :]

    #Construct Gaussian pyramid
    gaussian_video[buffer_index] = helper_fn.create_gaussian_pyramid(detection_frame, levels+1)[levels]
    fourier_transform = np.fft.fft(gaussian_video, axis=0)

    #Bandpass filter
    fourier_transform[mask == False] = 0

    #Grab a pulse
    if buffer_index % bpm_frequency == 0:

        i += 1

        for buffer in range(buffer_size):
            fourier_transform_avg[buffer] = np.real(fourier_transform[buffer]).mean()
        
        hz = frequencies[np.argmax(fourier_transform_avg)]
        bpm = 60.0 * hz
        bpm_buffer[bpm_buffer_index] = bpm
        bpm_buffer_index = (bpm_buffer_index + 1) % bpm_buffer_size

    #Amplify
    filtered = np.real(np.fft.ifft(fourier_transform, axis=0))
    filtered = filtered * alpha

    #Reconstruct resulting frame
    filtered_frame = helper_fn.reconstruction(filtered, buffer_index, levels, video_height, video_width)
    output_frame = detection_frame + filtered_frame
    output_frame = cv2.convertScaleAbs(output_frame)

    buffer_index = (buffer_index + 1) % buffer_size

    frame[video_height//2:total_height-video_height//2, video_width//2:total_width-video_width//2, :] = output_frame
    cv2.rectangle(frame, (video_width//2 , video_height//2), (total_width-video_width//2, total_height-video_height//2), box_color, box_weight)
    
    if i > bpm_buffer_size:
        cv2.putText(frame, "BPM: %d" % bpm_buffer.mean(), HR_text_location, font_style, font_scale, font_color)
    else:
        cv2.putText(frame, "Measuring Heart Rate...", init_text_location, font_style, font_scale, font_color)

    output_writer.write(frame)

    if len(sys.argv) != 2:
        cv2.imshow("Contactless Heart Rate Monitor", frame)

        #Exit if Key Q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #Exit if Esc is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

camera.release()
cv2.destroyAllWindows()
output_writer.release()


