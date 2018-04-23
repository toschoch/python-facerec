#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 23.04.2018
# author:  TOS

import logging
import uuid
import numpy as np
import copy
from collections import deque
from threading import RLock, Thread

from .utils import RepeatingTimer
from .dlib_api import detect_and_identify_faces
from .facedb import assert_session

import dlib

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class FaceTracker(object):
    """ Trackes Faces in an concurent stream of images. """
    def __init__(self, max_relative_shift=0.8, avg_over_nframes=1, missing_tolerance_nframes=0):

        self.max_rel_shift = max_relative_shift
        self.avg_over_nframes = avg_over_nframes
        self.missing_tol_nframes = missing_tolerance_nframes
        self.tracked_faces = {}
        self.detector = dlib.get_frontal_face_detector()
        self.last_frame = None

        self._verifier = RepeatingTimer(1.0, function=FaceTracker._regular_verification,args=(self.get_frame, self.get_tracked_faces, self.max_rel_shift))
        self._verifier.start()

    @staticmethod
    def _regular_verification(get_frame, get_tracked_faces, max_rel_shift):
        frame = get_frame()
        tracked_faces = get_tracked_faces()
        if frame is not None:
            FaceTracker._verify_identify(frame, tracked_faces, max_rel_shift)

    @staticmethod
    def _verify_identify(frame, tracked_faces, max_rel_shift):
        log.info("verify the frame")
        session = assert_session()
        persons = detect_and_identify_faces(frame, session)
        for person, rect, shapes in persons:
            face_coordinates = np.asarray(
                [rect.left(), rect.top(), rect.right(), rect.bottom()])

            for track_id, face in tracked_faces.items():
                if FaceTracker.is_same_face(face.coords(), face_coordinates, max_rel_shift):
                    del tracked_faces[track_id]
                    face.set(name=copy.copy(person.name))
                    break

    def verify(self, frame):
        self._verification_thread = Thread(target=FaceTracker._verify_identify, args=(frame, self.tracked_faces.copy(), self.max_rel_shift))
        self._verification_thread.start()

    @staticmethod
    def is_same_face(coord1, coord2, max_rel_shift_per_frame):
        x11, y11, x12, y12 = coord1
        x21, y21, _, _ = coord2

        w = abs(x12 - x11)
        h = abs(y12 - y11)

        return (abs(x11 - x21) / w < max_rel_shift_per_frame) and (abs(y11 - y21) / h < max_rel_shift_per_frame)

    def update(self, frame):

        current_faces_rects = self.detector(frame)

        faces_in_frame = []
        any_new_faces = False
        for face_rect in current_faces_rects:

            face_coordinates = np.asarray(
                [face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()])

            this_face = None
            for track_id, face in self.tracked_faces.items():
                if self.is_same_face(face.coords(), face_coordinates, self.max_rel_shift):
                    this_face = face
                    del self.tracked_faces[track_id]
                    break

            if this_face is None:
                # create new tracked face
                this_face = TrackedFace(face_coordinates)
                track_id = this_face.id()
                any_new_faces = True
            else:
                this_face.update_in_frame(face_coordinates)
            faces_in_frame.append((track_id, this_face))

        faces_recently_in_frame = []
        for track_id, face in self.tracked_faces.items():
            face.update_not_in_frame()
            if face._not_in_frame < self.missing_tol_nframes:
                faces_recently_in_frame.append((track_id, face))

        self.tracked_faces = dict(faces_in_frame + faces_recently_in_frame)

        self.last_frame = frame

        # if any_new_faces:
        #     self.verify(frame)

        return self.tracked_faces.values()

    def get_frame(self):
        try:
            return self.last_frame.copy()
        except AttributeError:
            return None

    def get_tracked_faces(self):
        return self.tracked_faces.copy()

class TrackedFace(object):
    """ a face tracked in video stream """

    def __init__(self, coords):

        self._tracker_id = uuid.uuid4()
        self._coords_buffer = deque(maxlen=5)
        self.update_in_frame(coords)

        self._face_date = {}
        self._lock = RLock()

    def set(self, **kwargs):
        self._lock.acquire(True)
        for key, value in kwargs.items():
            self._face_date[key] = value
        self._lock.release()

    def get(self, *args, **kwargs):
        self._lock.acquire(True)
        try:
            v = self._face_date.get(*args, **kwargs)
        finally:
            self._lock.release()
        return v

    def id(self):
        return self._tracker_id

    def coords(self):
        coords = np.mean(np.asarray(self._coords_buffer), axis=0).round().astype(int)
        return coords

    def update_in_frame(self, coords):
        self._not_in_frame = 0
        self._coords_buffer.append(coords)

    def update_not_in_frame(self):
        self._not_in_frame += 1


