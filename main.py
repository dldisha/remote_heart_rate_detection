#Importing libraries
import sys
import numpy as np
import cv2

#face detection using Haar cascades 
cascPath = "face_detection.xml"
#loading the cascade
faceCascade = cv2.CascadeClassifier(cascPath)

#import python file modules
import helper

#Show Webcam 
camera = None
#read the video
camera = cv2.VideoCapture(0) 

#Webcam parameters   
total_width = 640
total_height = 480
#Spatiotemporal box parameters
video_width = 320
video_height = 240
channels = 3
frame_rate = 12

#Setting the video frame
camera.set(15, total_width)
camera.set(10, total_height)
#camera.set(10, video_width)
#camera.set(8, video_height)

#Color Magnification Parameters
levels = 3
alpha = 150 # Amplifying factor
min_frequency = 1.0
max_frequency = 2.0
buffer_size = 120
buffer_index = 0

#Heart rate text display Parameters
font_style = cv2.FONT_HERSHEY_DUPLEX
font_scale = 1
font_color = (253, 3, 3)
init_text_location = (160, 35)
HR_text_location = (video_width//2 + 100, 35)
box_color = (5, 5, 5)
box_weight = 4

#Heart Rate calculation variables
bpm_frequency = 10
bpm_buffer_index = 2
bpm_buffer_size = 10
bpm_buffer = np.zeros((bpm_buffer_size))

#Saving the Output Heart Rate video as jpeg
output_file = "output.mov"
output_writer = cv2.VideoWriter()
output_writer.open(output_file, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), frame_rate, (total_width, total_height), True)

#Initializing the Gaussian Pyramid
first_frame = np.zeros((video_height, video_width, channels))
#Getting the first level
first_gaussian_pyramid = helper.create_gaussian_pyramid(first_frame, levels+1)[levels]
gaussian_video = np.zeros((buffer_size, first_gaussian_pyramid.shape[0], first_gaussian_pyramid.shape[1], channels))
#Taking Fourier Transform Average
fourier_transform_avg = np.zeros((buffer_size))

#Bandpass filter
frequencies = (1.0*frame_rate) * np.arange(buffer_size) / (1.0*buffer_size)
#Masking the special frequencies
mask = (frequencies >= min_frequency) & (frequencies <= max_frequency)


                                            ####### EULERIAN VIDEO MAGNIFICATION ALGORITHM #######

i = 0
HR = []

while (True):
    #Read the frame
    init, frame = camera.read()
    #frame = cv2.resize(frame, (600, 400))

    #converting into grayscale
    convert_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces using K-Nearest Neighbors
    face_detect = faceCascade.detectMultiScale(
        convert_gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    if init == False:
        break

    #When face is detected
    if len(face_detect) > 0:  

        detection_frame = frame[video_height//2:total_height - video_height//2, video_width//2:total_width - video_width//2, :]

        ### Calculating the heart rate ###

        #Constructing the Gaussian pyramid
        #Broadcasting input array shape
        gaussian_video[buffer_index] = helper.create_gaussian_pyramid(detection_frame, levels+1)[levels]
        #Transforming gaussian video using FFT
        fourier_transform = np.fft.fft(gaussian_video, axis=0)

        #Bandpass filter
        fourier_transform[mask == False] = 0

        #Finding a pulse
        if buffer_index % bpm_frequency == 0:
            i += 1

            for buffer in range(buffer_size):
                fourier_transform_avg[buffer] = np.real(fourier_transform[buffer]).mean()
            
            #Getting frequency in Hertz
            hz = frequencies[np.argmax(fourier_transform_avg)]
            #Calculating Heart Rate in Hz
            bpm = 60.0 * hz
            bpm_buffer[bpm_buffer_index] = bpm
            bpm_buffer_index = (bpm_buffer_index + 1) % bpm_buffer_size

        #Amplifying FFT by a factor of alpha (150)
        filtered = np.real(np.fft.ifft(fourier_transform, axis=0))
        filtered = filtered * alpha

        #Reconstructing the resulting frame
        filtered_frame = helper.reconstruction(filtered, buffer_index, levels, video_height, video_width)
        #Combing the detected and filtered frame 
        output_frame = detection_frame + filtered_frame
        output_frame = cv2.convertScaleAbs(output_frame)

        buffer_index = (buffer_index + 1) % buffer_size

        frame[video_height//2:total_height-video_height//2, video_width//2:total_width-video_width//2, :] = output_frame
        cv2.rectangle(frame, (video_width//2 , video_height//2), (total_width-video_width//2, total_height-video_height//2), box_color, box_weight)
        
        #Showing Heart Rate in BPM
        if i > bpm_buffer_size:
            cv2.putText(frame, "BPM: %d" % bpm_buffer.mean(), HR_text_location, font_style, font_scale, font_color)
            HR.append(bpm_buffer.mean())

        #Loading the Heart rate  
        else:
            cv2.putText(frame, "Measuring Heart Rate...", init_text_location, font_style, font_scale, font_color)

        #Writing into output file
        output_writer.write(frame)

        if len(sys.argv) != 2:
            #Display
            cv2.imshow("Remote Heart Rate Monitor", frame)

            #Exit if Key Q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #Exit if Esc is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break

    #When No face is detected
    else:
        #No face detected
        cv2.putText(frame, "No face detected", init_text_location, font_style, font_scale, font_color)
        #Writing it into output video
        output_writer.write(frame)
        
        if len(sys.argv) != 2:
            #Display
            cv2.imshow("Remote Heart Rate Monitor", frame)
            
            #Exit if Key Q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #Exit if Esc is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break

#Release the Camera object
camera.release()
#Destroy all windows when Esc or Q pressed
cv2.destroyAllWindows()
#Save the output video into the folder
#Note: it rewrites the output file on every program run
output_writer.release()

#Average of the first 30 buffers
print("Average Heart rate:", sum(HR[:30])/30)
hr = sum(HR[:30])/30