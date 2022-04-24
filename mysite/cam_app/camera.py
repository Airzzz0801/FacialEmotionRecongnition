import pickle
from django.conf import settings
from cam_app import views
from django.http import StreamingHttpResponse
import sqlite3
import datetime

# import some common libraries
import numpy as np
import os, json, cv2, random, glob, uuid
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
import time
from cam_app import predictModel

class VideoCamera(object):


    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame_with_detection(self):
        if predictModel.model == None:
            predictModel.load_model()
        success, image = self.video.read()
        labelNames = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        outputs = image
        # if you dont want to show the detection, comment the below code till outputImage = image, and change it to outputImage = outputs
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        # eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # gray = cv2.equalizeHist(gray)
        #-- Detect faces
        # faces = face_cascade.detectMultiScale(gray)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1,minSize=(120,120))
        for (x,y,w,h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face_arr = face.astype(np.float32)
            face_arr /= 255.
            face_arr = np.expand_dims(face_arr, axis=0)
            predictions = predictModel.model.predict(face_arr)

            center = (x + w//2, y + h//2)
            image = cv2.ellipse(image, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            i = np.argmax(predictions, axis=1)
            cv2.putText(image,labelNames[i[0]],(x, y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4,8)
            # faceROI = gray[y:y+h,x:x+w]
            #-- In each face, detect eyes
            # eyes = eyes_cascade.detectMultiScale(faceROI)
            # for (x2,y2,w2,h2) in eyes:
            #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            #     radius = int(round((w2 + h2)*0.25))
            #     image = cv2.circle(image, eye_center, radius, (255, 0, 0 ), 4)
        outputImage = image
        ret, outputImagetoReturn = cv2.imencode('.jpg', outputImage) # check if it work
        return outputImagetoReturn.tobytes(), outputImage

def generate_frames(camera):
    try:
        while True:
            frame, img = camera.get_frame_with_detection()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        print(e)

    finally:
        print("Reached finally, detection stopped")
        cv2.destroyAllWindows()
