import json
import datetime
import numpy as np
from PIL import Image, ImageOps
import os
from scipy import ndimage, misc

class package:
    def __init__(self,frame,senderCameraId):
        self.frame= frame
        self.sendingTime = datetime.datetime.now().__str__()
        self.senderCameraId = senderCameraId

#######To convert our object into a json object inorder to send it to another computer
####### we use .__dict__ because json only recognizes built-in types, so any other class won't be serialized
    def serialize_package(self):
         return json.dumps(self.__dict__)

######this methode is to unpack (deserialize) the package
    def deserialize_package(self,serializedPackage):
        return json.loads(serializedPackage)


##### this methode is for extracting frames,grayscale it, crop it, and save it
def extract_crop_frame(frame,faces,number):
    rescale = (255.0 / frame.max() * (frame - frame.min())).astype(np.uint8)
    size = (160,160)
    ####convert the image to grayscale numpy array
    for facet in faces:

        image = Image.fromarray(rescale)#.convert('LA')
        #####extracting the boundings to
        boundings = facet.bounding_box.astype(int)
        image = image.crop((boundings[0], boundings[1], boundings[2], boundings[3]))

        # score = cv2.Laplacian(image, cv2.CV_64F).var()
        # image = image[boundings[0],boundings[3]]
        # if score >threshold:
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image.save('tmp/'+ str(number) + 'test.png')

#just in case,after the fixes i made i won't need it
'''
def converg_rgb_grayscale(path):
    for images in os.listdir(path):
        print('images are')
        print(images)
        grayscaleImage = np.dot(np.asanyarray(Image.open(os.path.join(path,images)))[...,:3], [0.2989, 0.5870, 0.1140])
        print(grayscaleImage.shape)

        misc.imsave(os.path.join(path,images), grayscaleImage)'''





