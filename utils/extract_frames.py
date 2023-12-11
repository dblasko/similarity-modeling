import pandas as pd
import math
import cv2
from tqdm import tqdm
import os
cwd = os.getcwd()

data_path = "../data/"

#episodes should be split into training, test & validation data
training_frames_path = data_path + "training_frames/"
test_frames_path = data_path + "test_frames/"
validation_frames_path = data_path+"validation-frames/"

if not os.path.exists(training_frames_path):
    os.makedirs(training_frames_path)

if not os.path.exists(test_frames_path):
    os.makedirs(test_frames_path)

if not os.path.exists(validation_frames_path):
    os.makedirs(validation_frames_path)



def save_frames(file, i):
    video_path = f"{data_path}{file}avi"
    csv_path = f"{data_path}{file}csv"
    print("Saving frames from "+video_path +"video"+str(i+1))
   

    capture = cv2.VideoCapture(f"{data_path}{f}avi")
    success, frame = capture.read()
    pbar = tqdm(total=capture.get(7))
    frameNr = 0

    while success:
        #training data
        if i == 0:
            cv2.imwrite(f'{training_frames_path}frame_{frameNr}.png',frame)
        #test data
        if i == 1:
            cv2.imwrite(f'{test_frames_path}frame_{frameNr}.png',frame)
        #validation data
        if i == 2:
            cv2.imwrite(f'{validation_frames_path}frame_{frameNr}.png',frame)


        success, frame = capture.read()
        frameNr = frameNr + 1
        pbar.update(1)

    capture.release()

    print("all frames have been extracted successfully")


files = ["Muppets-02-01-01.", "Muppets-02-04-04.", "Muppets-03-04-03."] 

for i, f in enumerate(files):
    save_frames(f, i)
