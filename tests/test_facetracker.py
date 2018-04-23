#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 23.04.2018
# author:  TOS

import logging
import os
import cv2
from facerec.facetracker import FaceTracker

log = logging.getLogger(__name__)

here = os.path.split(__file__)[0]

def test_webcamstream():

    tracker = FaceTracker()
    cam = cv2.VideoCapture(0)
    color_green = (0, 255, 0)
    line_width = 3

    while True:
        ret_val, img = cam.read()
        faces = tracker.update(img)
        for face in faces:
            coords = face.coords();
            cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), color_green, line_width)
            cv2.putText(img, face.get('name','unknown'), (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()