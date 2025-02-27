# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


#################Important: Colors of the bouding box depends on whether we recognize the person or not, Blue if positive
################# Red if Négative
#################Performence according to Frame intervale On Asus Rogue I5 7300-HQ @2.5GHz Nvidia GTX 1050
#################Frame_interval = 2 =====> 9FPS
#################Frame_interval = 3 =====> 7FPS
#################Frame_interval = 5 =====> 9FPS

import argparse
import sys
import time
import numpy as np
import cv2
import face
from PIL import Image
import send_package as sp
import pika

############# propriétés du bouding box (ma touchihach mohamed)
def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          face.color, 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_COMPLEX, 1, face.color,
                            thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,51),
                thickness=2, lineType=2)



######Get the name of each face detected
def get_name(faces):
    if faces is not None:
        for face in faces:
             return face.name



def main(args):
    frame_interval =1   # Number of frames after which to run face detection
    fps_display_interval = 1  # seconds
    frame_rate = 0
    frame_count = 0
    imageNumber= 0
    video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition()
    start_time = time.time()

    credentials = pika.PlainCredentials('mak','26121997')
    HOST ='localhost'
    ######Creating communication channel between 2 computers
    connexion = pika.BlockingConnection(pika.ConnectionParameters(host=HOST,credentials=credentials))
    channel=connexion.channel()
    channel.queue_declare(queue="messages")


    if args.debug:
        print("Debug enabled")
        face.debug = True

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        #threshold = 110 ####this part is for comparing the blur
        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            #print(type(faces))
            #print(len(faces)) #this code is just for testing face object
            #--------------------------------------------------cuthere--------------------------------------------------
            ####this part is for extracting the frame and converting it to rgb

           #------------------------------------------------------------------------------------------------------------


            #####get the current face name
            for f in faces:
                if(f is not None and f.name == 'Intrus'):
                    #####we create a new package object and serialize it
                    sp.extract_crop_frame(frame, faces, imageNumber)
                    imageNumber += 1
                    sealed_package = sp.package(frame.tolist(),'0').serialize_package()
                    channel.basic_publish(exchange='',routing_key='messages',body=sealed_package)
                    print('The sealed package is sent')




            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate)


        frame_count += 1
        cv2.imshow('Video', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    connexion.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
