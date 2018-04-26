#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 23.04.2018
# author:  TOS

import pytest
from .test_facedb import tmpdb
import glob
import os
import pathlib
import cv2
import facerec.dlib_api
import facerec.facedb

here = os.path.split(__file__)[0]

def test_identify_no_face(tmpdb):
    img = cv2.imread(os.path.join(here, 'data', "Tobias_Schoch_TOS_big (Large).jpg"))
    persons = facerec.dlib_api.detect_and_identify_faces(img)

def test_teach_tos(tmpdb):
    for f in glob.glob(os.path.join(here, 'data', "*.jpg")):
        f = pathlib.Path(f)
        if f.name.startswith('WIN_') or f.name.startswith('2018-'):
            #print("Processing file: {}".format(f))
            img = cv2.imread(str(f))
            facerec.dlib_api.teach_person(img, "Tobias Schoch")

    persons = facerec.facedb.persons()
    assert len(persons)==1
    assert persons[0].name == "Tobias Schoch"

    img = cv2.imread(os.path.join(here, 'data', "Tobias_Schoch_TOS_big (Large).jpg"))
    persons = facerec.dlib_api.detect_and_identify_faces(img)
    assert len(persons)==1
    assert persons[0][0].name == "Tobias Schoch"

    img = cv2.imread(os.path.join(here, 'data', r"David_Fries_FDA_big (Large).jpg"))
    persons = facerec.dlib_api.detect_and_identify_faces(img)
    assert len(persons) == 1
    person = persons[0][0]
    print(person.id)
    print(person.name)