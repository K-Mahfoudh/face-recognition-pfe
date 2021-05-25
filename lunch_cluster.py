import tensorflow as tf
import numpy as np
import importlib
import argparse
import scipy
import facenet
import os
import math
import cv2 as cv
from os.path import join, basename, exists
from os import makedirs
import numpy as np
import shutil
import send_package as sp
import treat_clusters as tc
import cv2
import classifierModified as cm
from PIL import Image
import intruderFace as f
import time






def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    import numpy as np
    if len(face_encodings) == 0:
        return np.empty((0))

    # return 1/np.linalg.norm(face_encodings - face_to_compare, axis=1)
    return np.sum(face_encodings * face_to_compare, axis=1)


def load_model(model_dir, meta_file, ckpt_file):
    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))


def _chinese_whispers(encoding_list, threshold=0.55, iterations=20):
    """ Chinese Whispers Algorithm

    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/

    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate

    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """

    # from face_recognition.api import _face_distance
    from random import shuffle
    import networkx as nx
    # Create graph
    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        print("No enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx + 1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
        nodes.append(node)

        # Facial encodings to compare
        if (idx + 1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx + 1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):

            if distance > threshold:
                # Add edge if facial match
                edge_id = idx + i + 2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = list(G.nodes())
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if G.node[ne]['cluster'] in clusters:
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            # use the max sum of neighbor weights class as current node's class
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():
        cluster = data['cluster']
        path = data['path']

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters


def cluster_facial_encodings(facial_encodings):
    """ Cluster facial encodings

        Intended to be an optional switch for different clustering algorithms, as of right now
        only chinese whispers is available.

        Input:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

        Output:
            sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
                sorted by largest cluster to smallest

    """

    if len(facial_encodings) <= 1:
        #print("Number of facial encodings must be greater than one, can't cluster")
        return []

    # Only use the chinese whispers algorithm for now
    sorted_clusters = _chinese_whispers(facial_encodings.items())
    return sorted_clusters


def compute_facial_encodings(sess, images_placeholder, embeddings, phase_train_placeholder, image_size,
                             embedding_size, nrof_images, nrof_batches, emb_array, batch_size, paths):
    """ Compute Facial Encodings

        Given a set of images, compute the facial encodings of each face detected in the images and
        return them. If no faces, or more than one face found, return nothing for that image.

        Inputs:
            image_paths: a list of image paths

        Outputs:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

    """

    for i in range(nrof_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

    facial_encodings = {}
    for x in range(nrof_images):
        facial_encodings[paths[x]] = emb_array[x, :]

    return facial_encodings


def get_onedir(paths):
    dataset = []
    path_exp = os.path.expanduser(paths)
    if os.path.isdir(path_exp):
        images = os.listdir(path_exp)
        image_paths = [os.path.join(path_exp, img) for img in images]

        for x in image_paths:
            if os.path.getsize(x) > 0:
                dataset.append(x)

    return dataset


def main(args):
    tmp_file_path="./tmp"
    repeate_dict = {} #to check the number of times a certain image was clustered
    repeated_images_list=[] #this list is for storing images paths of a cluster with length <10 to check if the imges are repeated or not
    loading_start_time = time.time() #==================================================================================
    with tf.Graph().as_default():
        with tf.Session() as sess:
            loaded = False
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser('./models'))
            load_model('./models', meta_file, ckpt_file)
            if tc.trained_model_exists("./myclassifier/my_intruder_classifier.pkl"):
                intruder_face_recognition = f.Recognition()
                loaded = True
            loading_time = time.time()-loading_start_time #=============================================================
            while True:
             # image_list, label_list = facenet.get_image_paths_and_labels(train_set)
                image_paths = get_onedir('./tmp')
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                image_size = images_placeholder.get_shape()[1]
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                embeddings_start_time = time.time() #===================================================================
                if len(image_paths) >15:
                    print('Runnning forward pass on images')
                    nrof_images = len(image_paths)
                    nrof_batches = int(math.ceil(1.0 * nrof_images / args.batch_size))
                    emb_array = np.zeros((nrof_images, embedding_size))
                    facial_encodings = compute_facial_encodings(sess, images_placeholder, embeddings,
                                                            phase_train_placeholder, image_size,
                                                            embedding_size, nrof_images, nrof_batches, emb_array,
                                                            args.batch_size, image_paths)
                    embeddings_time = time.time() - embeddings_start_time  # ===============================================
                    clustering_start_time = time.time() ##==================================================================
                    sorted_clusters = cluster_facial_encodings(facial_encodings)
                    clustering_time = time.time()-clustering_start_time #===================================================

                    print(sorted_clusters)
                    print(type(sorted_clusters))
                    num_cluster = len(sorted_clusters)

                    # Copy image files to cluster folders
                    for _, cluster in enumerate(sorted_clusters):
                        # save all the cluster
                        #print(cluster)

                        fileNumber = len(os.listdir('./clusters'))

                        cluster_dir = join('./clusters', 'cluster '+str(fileNumber))
                        #we save the cluster (if it has at least 10 images) in order to train it later
                        if len(cluster) >= 10:
                            if not exists(cluster_dir):
                                makedirs(cluster_dir)
                            for path in cluster:

                                shutil.move(path, join(cluster_dir, basename(path)))
                                print("we have moved this file to {}".format(join(cluster_dir, basename(path))))
                                #each time we store an image, we remove it from the tmp folder to avoid clustering it again
                                #os.remove(path)

                    ###Clusters treating time measuring
                    treating_start_time = time.time()
                    #Saving images that were not clustered, in order to check if they are trush images (to delet them later)

                    for path in os.listdir(tmp_file_path):
                        print(path)
                        # each time we find a cluster with less than 10 images, we add its images paths to a list
                        if not path in repeate_dict:

                            # repeated_images_list.append(path)
                            repeate_dict.update(
                                {path: 1})  # this dictionary is to check number of times we did clustering
                            # on a certain image (we init it to 1)
                        else:  # in this case, the image was already clustered at least once
                            repeate_dict[path] += 1
                            if repeate_dict[path] == 3:  # clustered image 3 times at least, so it doesn't belong to an intruder
                                if (os.path.exists(os.path.join("./tmp", path))):
                                    os.remove(os.path.join("./tmp", path))
                                    print('we removed %s' % path)
                                    del repeate_dict[path]

                    # checking if the intruder already exists or not
                    #Write your code here
                    for folders in os.listdir('./clusters/'):
                        ####this is the additional code
                        recognized = 0
                        unregognized = 0
                        index = 0
                        Path = os.path.join('./clusters/',folders)
                        intruderImages = os.listdir(os.path.join('./clusters/',folders))
                        print("we are in cluster============================================================>")
                        print(Path)
                        if (os.path.exists(Path)):
                            imagesNumber = len(intruderImages)
                            if loaded == True: ###It means that the classifier already exists
                                ###The classifier exists means that we have detected 1 intruder at least, and trained our model at least once
                                while index < imagesNumber and recognized <= imagesNumber / 2 and unregognized <= imagesNumber / 2:

                                    im = cv2.imread(os.path.join(Path, intruderImages[index]))


                                    prediction = intruder_face_recognition.identify(im)
                                    for face in prediction:

                                        print(face.name)
                                        if face.name == 'unrecognized':
                                            unregognized += 1
                                        else:
                                            recognized += 1
                                    index += 1

                                print("recognized are %d and unrecognized are %d" % (
                                        recognized, unregognized))
                                if recognized > unregognized:
                                    print("We will remove the cluster {}".format(folders))
                                    shutil.rmtree(Path)
                                else:
                                    print("intruder detected,")
                                    shutil.move(Path, "./training_zone/cluster_{}".format(str(len(os.listdir("./training_zone")))))
                                    ###running training process
                                    cm.main(cm.parse_arguments(cm.sys.argv[1:]))
                                    ###Reloading the new model
                                    intruder_face_recognition = f.Recognition()
                            else:
                                print("intruder detected,")
                                shutil.move(Path, "./training_zone/cluster_{}".format(str(len(os.listdir("./training_zone")))))
                                ###running training process
                                cm.main(cm.parse_arguments(cm.sys.argv[1:]))
                                ###Reloading the new model
                                intruder_face_recognition = f.Recognition()
                    treating_time = time.time() - treating_start_time


    print("Loading time: {}\nfeature calculating time: {}\nClustering time: {}\nTreating time: {}".format(loading_time,embeddings_time,clustering_time,treating_time))




#def check_intruder_existance:


def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
    parser.add_argument('--batch_size', type=int, help='batch size', default=30)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """ Entry point """
    main(parse_args())