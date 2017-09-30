"""FaceValid core code"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
#import facial_landmarks as fl
import cv2
from os import listdir
from os.path import isfile, join
import imghdr
import time
import pickle
from sklearn.svm import SVC


def loadModel(args):
    print('Creating Mtcnn networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False,device_count={"CPU": 2},
                #inter_op_parallelism_threads = 1, 
                intra_op_parallelism_threads = 2))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    print('Creating Facenet networks and loading parameters')
    with tf.Graph().as_default():
        config = tf.ConfigProto(device_count={"CPU": 2},
                #inter_op_parallelism_threads = 1, 
                intra_op_parallelism_threads = 2)
        sess = tf.Session(config=config)
        # Load the model
        facenet.load_model(args.model,sess)        
        # Get input and output tensors
        images_placeholder = sess.graph.get_tensor_by_name("input:0")
        embeddings = sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
        fnet = lambda img,ifTrain : sess.run(embeddings, feed_dict={images_placeholder: img, phase_train_placeholder:ifTrain})
    return pnet,rnet,onet,fnet

def inference(args,pnet,rnet,onet,fnet):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor	
    #mypath = args.image_path

    # # Read image from database and inference
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # myImage_dict = {}
    # myImage = []
    # index = 0
    # for file in onlyfiles:
    #     isImage = None
    #     name = file
    #     file = mypath + '/' + file
    #     isImage = imghdr.what(file)
    #     if isImage != None:
    #         myImage.append(file)
    #         name = name.split('.')[0]
    #         myImage_dict[index] = name
    #         index+=1
    # images = load_and_align_data(myImage, args.image_size, args.margin, pnet, rnet, onet)
    # #feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    # emb_database = fnet(images,False)
    classifier_filename_exp = os.path.expanduser(args.classifier_filename)
    with open(classifier_filename_exp, 'rb') as infile:
    	(model, class_names) = pickle.load(infile,encoding='latin1')

    print('Loaded classifier model from file "%s"' % classifier_filename_exp)

    with open(args.name_dict) as f:
    	content = f.readlines()
    content = [x.strip() for x in content]
    print('Load Name Dictionary')
    #Start opencv Camera 640*480
    cap = cv2.VideoCapture(0)
	
    while(True):
        start = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        bounding_boxes, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)

        if bounding_boxes.shape[0] > 0:
            #finding face
            capture = align_single_data(frame, args.image_size, args.margin , bounding_boxes)
            #feed_dict = { images_placeholder: capture, phase_train_placeholder:False }

            emb_capture = fnet(capture,False)
            predictions = model.predict_proba(emb_capture)
            best_class_indices = np.argmax(predictions, axis=1)		 

            for i in range(emb_capture.shape[0]):          
                cv2.rectangle(frame, (int(bounding_boxes[i][0]), 
                int(bounding_boxes[i][1])), 
                (int(bounding_boxes[i][2]), int(bounding_boxes[i][3])), 
                (0, 255, 0), 2)
                #if closest < 0.8:
                cv2.putText(frame , content[best_class_indices[i]], (int(bounding_boxes[i][0]),int(bounding_boxes[i][3])), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255 ,0 ,0), 
                thickness = 1, lineType = cv2.LINE_AA)
                # else :
                #     cv2.putText(frame,'Warning Not in Database', (int(bounding_boxes[i][0]),int(bounding_boxes[i][3])), 
                #     cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 
                #     thickness = 2, lineType = 2)                        	
	    	
	    # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end = time.time()
        print(end-start)
	# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def align_single_data(image,image_size, margin, bounding_boxes):
    output = [None] * bounding_boxes.shape[0]

    for numFace in range(bounding_boxes.shape[0]):
        #img = misc.imread(os.path.expanduser(image))   
        img_size = np.asarray(image.shape)[0:2]
        det = np.squeeze(bounding_boxes[numFace,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1] - 1)
        bb[3] = np.minimum(det[3]+margin/2, img_size[0] - 1)
        #fix bbox error
        if bb[1] > bb[3]:
            bb[1] , bb[3] =  bb[3] , bb[1]
        if bb[0] > bb[2]:
            bb[0] , bb[2] =  bb[2] , bb[0]            
        cropped = image[bb[1]:bb[3],bb[0]:bb[2],:]
        #print(bb[0],bb[1],bb[2],bb[3])
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        output[numFace] = prewhitened
    

    return output   	

def load_and_align_data(image_paths, image_size, margin,  pnet, rnet, onet):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
  
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))   
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1] - 1)
        bb[3] = np.minimum(det[3]+margin/2, img_size[0] - 1)
        #fix bbox error
        if bb[1] > bb[3]:
            bb[1] , bb[3] =  bb[3] , bb[1]
        if bb[0] > bb[2]:
            bb[0] , bb[2] =  bb[2] , bb[0]   
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        #aligned = fl.faceLandMarks(aligned)
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)

    return images

# def compare_function():
#     nrof_images = len(imageDir)

#     print('Images:')
#     for i in range(nrof_images):
#         print('%1d: %s' % (i, imageDir[i]))
#     print('')
    
#     # cal  distance matrix
#     print('Distance  matrix')
#     print('    ', end='')
#     for i in range(nrof_images):
#         print('    %1d     ' % i, end='')
#     print('')
#     for i in range(nrof_images):
#         print('%1d  ' % i, end='')
#         for j in range(nrof_images):
#             dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
#             print('  %1.4f  ' % dist, end='')
#         print('')
    
#     # cal square distance matrix
#     print('Distance square matrix')
#     print('    ', end='')
#     for i in range(nrof_images):
#         print('    %1d     ' % i, end='')
#     print('')
#     for i in range(nrof_images):
#         print('%1d  ' % i, end='')
#         for j in range(nrof_images):
#             dist = np.sum(np.square(np.subtract(emb[i,:], emb[j,:])))
#             print('  %1.4f  ' % dist, end='')
#         print('')

#     # #cal cosine simularity
#     print('Consine Simularity matrix')
#     print('    ', end='')
#     for i in range(nrof_images):
#         print('    %1d     ' % i, end='')
#     print('')
#     for i in range(nrof_images):
#         print('%1d  ' % i, end='')
#         for j in range(nrof_images):
#             #dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
#             dist = np.sum(np.dot(emb[i,:], emb[j,:])) / (np.sqrt(np.sum(np.square(emb[i,:]))) * np.sqrt(np.sum(np.square(emb[j,:]))))
#             print('  %1.4f  ' % dist, end='')
#         print('')

#     cal correlation coefficient
#     mean_List = []
#     print('Correlation coefficient matrix')
#     print('    ', end='')
#     for i in range(nrof_images):
#         print('    %1d     ' % i, end='')
#         mean_List.append(np.mean(emb[i,:]))
#     print('')
#     for i in range(nrof_images):
#         print('%1d  ' % i, end='')
#         for j in range(nrof_images):
#             #dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
#             nu = np.sum(np.dot(np.subtract(emb[i,:],mean_List[i]) , np.subtract(emb[j,:],mean_List[j])))
#             de = np.sqrt(np.sum(np.square(np.subtract(emb[i,:],mean_List[i])))) * np.sqrt(np.sum(np.square(np.subtract(emb[j,:],mean_List[j]))))
#             dist = nu / de
#             print('  %1.4f  ' % dist, end='')
#         print('')