#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 20.04.2018
# author:  TOS

import logging
import dlib
import numpy as np
import cv2
import glob, os
import pkg_resources
from .facedb import Person

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(pkg_resources.resource_filename('facerec',r'models\shape_predictor_68_face_landmarks.dat'))
facerec = dlib.face_recognition_model_v1(pkg_resources.resource_filename('facerec',r'models\dlib_face_recognition_resnet_model_v1.dat'))

log = logging.getLogger(__name__)

def detect_faces(image):
    dets = detector(img, 1)

    persons =
    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #     k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)

        # Compute the 128D vector that describes the face in img identified by
        # shape.  In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people. Here we just print
        # the vector to the screen.
        facecode = np.asarray(facerec.compute_face_descriptor(img, shape))


if __name__ == '__main__':

    import facedb
    for f in glob.glob(os.path.join(r"D:\Users\TOS\Pictures\Camera Roll", "*.jpg")):
        print("Processing file: {}".format(f))
        img = cv2.imread(f)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        # Now process each face we found.
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)


            # Compute the 128D vector that describes the face in img identified by
            # shape.  In general, if two face descriptor vectors have a Euclidean
            # distance between them less than 0.6 then they are from the same
            # person, otherwise they are from different people. Here we just print
            # the vector to the screen.
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            facedb.teach(face_descriptor, "Tobias Schoch")

