import os
import face
import shutil
import cv2
import numpy as np
from PIL import Image


if os.path.exists('./myclassifier/my_classifier.pkl') :
    intruder_face_recognition = face.Recognition()
#checks if a classifier already exists or not
def trained_model_exists(path):
    return os.path.exists(path)

#get the number of directories in a cluster
def number_of_clusters(path):
    return len(os.listdir(path))

# checks if the intruder exists in intruder list or not by calculating number of identified images (for more accuracy)
'''def intruder_exists(intruderFolder,classifierPath):
    recognized = 0
    unregognized = 0
    index =0
    intruderImages = os.listdir(intruderFolder)
    imagesNumber= len(intruderImages)
    if trained_model_exists(classifierPath):
        while index < imagesNumber and recognized <= imagesNumber/2 and unregognized<= imagesNumber/2:
            print(os.path.join(intruderFolder, intruderImages[index]))
            #image = np.asanyarray(cv2.imread(os.path.join(intruderFolder,intruderImages[index])))
            im = np.asanyarray(Image.open(os.path.join(intruderFolder,intruderImages[index])))
            print(type(im))
            prediction = intruder_face_recognition.identify(im)
            for face in prediction:
                if face.name == 'unrecognized':
                    unregognized +=1
                else:
                    recognized +=1
            index+=1
            print("recognized are %d and unrecognized are %d" %(recognized,unregognized))
    return True if recognized> unregognized else False

#i will modify this later in order to connect it to the database and send notifications
def send_notification(intruderFolder,classifierFolder):
    if( not intruder_exists(intruderFolder,classifierFolder)):
        print("Intruder notified")

def move_intruder_trainzone(currentFolder,trainingFolder,classifierPath):
    for folder in os.listdir(currentFolder):
        if intruder_exists(os.path.join(currentFolder,folder),classifierPath):
            shutil.rmtree(folder)
        else:
            shutil.move(os.path.join(currentFolder,folder),trainingFolder)'''






