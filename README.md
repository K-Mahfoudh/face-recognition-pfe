# face_recognition
real time face recognition with MTCNN and FaceNet

## Project description
The main goal of this project is the detection of intruders in a certain organism using one or multiple cctv cameras, it's based on FaceNet implementation and uses VGGFace2 pretrained model.
The system has 2 main tasks, face detection and recognition, and sending notification in case of intruder's detection, these 2 tasks are done in separate computers in order to have a faster result, and reduce the amount of processing each computer does.
## Before run code

you need to do things below:

*  I have already uploaded det1.npy det2.npy det3.npy which for MTCNN,but you still need to download facenet's pb file from [davidsandberg's
github](https://github.com/davidsandberg/facenet) like 20170511-185253,extract to pb file and put in models directory.
* tensorflow-gpu 1.1.0 , later version may also work.
* python 3.X


## Inspiration
* [davidsandberg's github](https://github.com/davidsandberg/facenet)

