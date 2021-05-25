"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
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



####This file aims to train intruders recognition model, the classifier is modified in order to make training faster
####when we calculate embeddings, we strore them in a file, and then we load them in the next time
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC

def main(args):
    mode = 'TRAIN'
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset('./training_zone')
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (mode=='TRAIN'):
                    dataset = train_set
                    print(len(dataset))
                elif (mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset('./training_zone')

            # Check that there are at least one training image per class
            for cls in dataset:
                assert len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset'
            old_file_number = 0
            if os.path.exists("last_class.txt") and os.path.exists('./myclassifier/my_intruder_classifier.pkl') :
                with open("last_class.txt", "r") as f:
                    old_file_number = int(f.read())
            with open("last_class.txt", 'w+') as f:
                f.write("%d" % len(dataset))
            dataset = dataset[old_file_number:]
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print(labels)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            ###saving the number of folders in our training folder in order to load it later


            # Load the model
            if len(dataset)>0:
                print('Loading feature extraction model')
                facenet.load_model('./models/20180402-114759.pb')
            
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]


                ###This is the part to modify, we should add a condition to check wether embedings file exist or not
                #Run forward pass to calculate embeddings
                print('Calculating features for images')
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i*args.batch_size
                    end_index = min((i+1)*args.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, args.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)




#####====================================================================================================================####
                ####At this point we save the embeddings and labels in a file
                ####The main goal of the file is loading previous embeddings and labels, without recalculating them
                classifier_filename_exp = os.path.expanduser('./myclassifier/my_intruder_classifier.pkl')
                if os.path.exists('./myclassifier/my_intruder_classifier.pkl'):
                    print("loading old features")
                    old_labels,old_emb_array = save_or_load_previous_data("emb.txt","labels.txt",labels,emb_array,"load")
                    print("The length of the old array is".format(len(old_emb_array)))

                    ###Saving the labels and embeddings in files in order to load them later
                    save_or_load_previous_data("emb.txt", "labels.txt", labels, emb_array, "save")

                    ###Adding new embedding to old embeddings array
                    print("the length of old Embedding array is {}".format(len(old_emb_array)))
                    if(len(old_emb_array)> 0):
                        emb_array = np.vstack([old_emb_array,emb_array])
                    print("the length of new Embeddings array is {}".format(len(emb_array)))

                    ###Adding new labels to old labels
                    print("the old label list was {}".format(old_labels))
                    print("==============={}".format(old_file_number))
                    labels = old_labels + [x + old_file_number for x in labels]
                    print("the new label list is {}".format(labels))
                else:
                    save_or_load_previous_data("emb.txt", "labels.txt", labels, emb_array, "save")


            if (mode=='TRAIN') and len(dataset)>0:
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
            
                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            elif (mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)
                
def save_or_load_previous_data(embFileName,labelFileName,labels,emb_array,operation):

    if(operation == 'save'):
        ####Saving data in files in order to load them later
        last_label_file = open(labelFileName, 'a')
        emb_file = open(embFileName,'ab')
        ####Writing in labels file
        for items in labels:
            last_label_file.write('%s\n' % items)

        ####Writing in embeddings file
        np.savetxt(emb_file, emb_array)
        last_label_file.close()
        emb_file.close()
    elif(operation == 'load'):
        print('loading')
        with open(embFileName,'r') as emb:
            embeddings = np.loadtxt(embFileName)

        with open(labelFileName,'r') as label:
            aList = []
            label.seek(0)
            for lines in label:
                if lines[:-1] != '':
                    aList.append(int(lines[:-1]))
        return aList,embeddings

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    '''parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
     parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.',default='./outputs/')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',default='./models/20180402-114759.pb')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.',default='./classifier/my_classifier.pkl')'''
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=50)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=10)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
