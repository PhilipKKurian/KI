#Imports
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import pathlib
import datetime
import os

###Functions

#Process video
def process_video(video_path):
    #open video file
    vc = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not vc.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get the frames per second (fps) and total number of frames of the video
    fps = vc.get(cv2.CAP_PROP_FPS)
    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = [] 

    #take a frame every second and add to list (analysing every single frame is not necessary as vehicles are visible in multiple frames) 
    i = 1
    while(i<=total_frames):
        # Set the video frame position
        vc.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        # Read the frame
        success, frame = vc.read()

        if success:
            # add the frame to the list
            frames.append(frame)
        else:
            print("Error: Could not read frame.")

        i=i+fps

    # Release the video capture object
    vc.release()

    #return the captured frames (output of this function)
    return(frames)


#Detect vehicles
#this function processes each frame, detects vehicles, and returns `True` if the specified vehicle is detected.
def detect_vehicles(frame, v_type, input_size):
   
    #resize image
    image = cv2.resize(frame, input_size)
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize the image
    
    #classification
    prediction = c_model.predict(image)
    predicted_class = list(prediction[0]).index(max(prediction[0]))
    confidence = max(prediction[0])

    if(predicted_class == v_type and confidence > 0.5): #confidence might need to be adjusted
        return True
    else:
        return False


#Save frame
def save_frame(frame, path, filename):
    cv2.imwrite(path+'/'+filename, frame)


###Model
program_path = str(pathlib.Path(__file__).parent.resolve()).replace("\\", "/")
#Load Classification Model
c_model = tf.keras.models.load_model(program_path+'/vehicle_classifier.h5')
c_labels = ['bus', 'car', 'no vehicle', 'truck', 'bicycle', 'motorbike', 'tractor']
input_size = (150, 150) #size of images the model can interpret


###Variables

#Get user input
print("Welcome to the vehicle detection prototype!\n\n")
print("This programm can detect the following vehicle types: bus, car, truck, bicycle, motorbike, tractor.\n")
vehicle_type = input("What type of vehicle would you like to detect?\n")
while(vehicle_type.lower() not in c_labels):
    vehicle_type = input("The provided vehicle type is not recognised. What type of vehicle would you like to detect?\n")
video_path = input("\n\nPlease enter the full path (path+filename) of the video you would like to analyse.\n")
print("\n\nThe video is being analysed. Please wait a moment.")

#Process user input
v_type = c_labels.index(vehicle_type.lower())

#Other variables
output_path = program_path + '/' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
if not os.path.exists(output_path):
    os.makedirs(output_path)


###Execute Vehicle Detection Program

#Extract frames from video
frames = process_video(video_path)

i=1
#Save frame if vehicle type is detected
for f in frames:
    if(detect_vehicles(f, v_type, input_size)):
        save_frame(f, output_path, vehicle_type+str(i)+".jpg")
        i+=1

#Inform user
print("\n\n"+str(i-1)+" vehicles of type "+vehicle_type+" have been detected in the video.")
print("The images have been saved to " + output_path)


###End program

#exit
end = input("\n\nPress enter to end the program...")
