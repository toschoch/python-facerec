#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 20.04.2018
# author:  TOS

import logging
import dlib
import numpy as np
import pkg_resources

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(pkg_resources.resource_filename('facerec',r'models/shape_predictor_68_face_landmarks.dat'))
facerec = dlib.face_recognition_model_v1(pkg_resources.resource_filename('facerec',r'models/dlib_face_recognition_resnet_model_v1.dat'))

log = logging.getLogger(__name__)

def detect_faces(image):
    """
    detects faces in image given and identifies the persons corresponding to the faces.
    Args:
        image: (np.array, cv.array) image given by 3d int array.

    Returns:
        list of 3-tuple, 128D-array (face codes, rect, shape)

    """
    dets = detector(image, 1)

    persons = []
    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #     k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(image, d)

        # Compute the 128D vector that describes the face in img identified by
        # shape.  In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people.
        facecode = np.asarray(facerec.compute_face_descriptor(image, shape))
        persons.append((facecode, d, shape))

    return persons

def detect_and_identify_faces(image, session=None):
    """
    detects faces in image given and identifies the persons corresponding to the faces.
    Args:
        image: (np.array, cv.array) image given by 3d int array.

    Returns:
        list of 3-tuple (Person, rect, shape)

    """

    from .facedb import identify_person

    dets = detector(image, 1)

    persons = []
    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #     k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(image, d)

        # Compute the 128D vector that describes the face in img identified by
        # shape.  In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people.
        facecode = np.asarray(facerec.compute_face_descriptor(image, shape))
        persons.append((identify_person(facecode, session=session), d, shape))

    return persons

def teach_person(image, name=None, id=None, weight=1.0, session=None):
    """
    teach the face recognition system that the given image containes a exactly one face of specified person.
    Args:
        image: (np.array, cv.array) image given by 3d int array.
        name: (str) name of the person in the face database (either name or id has to be specified)
        id: (int) id of the person in the face database (either name or id has to be specified)
        weight: (float, default 1.0) weight to assign to specific face code in teaching.

    Returns:
        3-tuple (Person, rect, shape)

    """

    from .facedb import teach

    dets = detector(image, 1)
    if len(dets) > 1:
        raise ValueError('More than one face detected on the passed image! An image with exactly one face has to be passed!')
    elif len(dets) == 0:
        raise ValueError('No face detected on the passed image! An image with exactly one face has to be passed!')
    else:
        shape = sp(image, dets[0])
        facecode = np.asarray(facerec.compute_face_descriptor(image, shape))
        person = teach(facecode, name, id, weight=weight, session=session)
        return (person, dets[0], shape)

