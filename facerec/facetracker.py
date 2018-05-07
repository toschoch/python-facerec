#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 23.04.2018
# author:  TOS

import logging
import uuid
import numpy as np
import copy
from collections import deque
from multiprocessing import Process, Manager, Event

from .dlib_api import detect_and_identify_faces, detect_faces
from .facedb import assert_session
from .client import FacerecApi

import dlib

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class Identifier(Process):
    def __init__(self, interval, function, args=[], kwargs={}):
        super(Identifier, self).__init__()
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.finished = Event()

    def cancel(self):
        """Stop the timer if it hasn't finished yet"""
        self.finished.set()

    def run(self):
        self.finished.wait(self.interval)
        if not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.run()

class FaceTracker(object):
    """ Trackes Faces in an concurent stream of images. """
    def __init__(self, url=None, max_relative_shift=0.8, avg_over_nframes=1, missing_tolerance_nframes=0):
        """
        creates a face tracker that identifies the tracked face with facerec engine. The requests to identify can be done locally
        or sent to a facerec server by specifying the url.
        Args:
            url: (optional) url of the facerec server
            max_relative_shift: (float) maximal shift of face from one frame to the other relative to face width (to be tracked as the same face)
            avg_over_nframes:  (int) optional moving average filter of the face position
            missing_tolerance_nframes: (int) how many frames should the tracker store a face that is not detected anymore (helps for short missing detection)
        """

        self.max_rel_shift = max_relative_shift
        self.avg_over_nframes = avg_over_nframes
        self.missing_tol_nframes = missing_tolerance_nframes

        self.detector = dlib.get_frontal_face_detector()

        self.memory_manager = Manager()
        self._shared = self.memory_manager.dict()
        self._shared['tracked_faces'] = self.memory_manager.dict()
        self.tracked_faces = {}

        if url is not None:
            self.api = FacerecApi(url)
            self._identifier = Identifier(1.0, FaceTracker._verify_identify_server,
                                          args=(self._shared, self.api, self.max_rel_shift))
        else:
            self.api = None
            self._identifier = Identifier(1.0, FaceTracker._verify_identify_local,
                                          args=(self._shared, self.max_rel_shift))

        self._identifier.start()


    @staticmethod
    def _verify_identify_server(shared, api, max_rel_shift):
        faces = detect_faces(shared['frame'])
        for facecode, rect, shapes in faces:
            face_coordinates = np.asarray(
                [rect.left(), rect.top(), rect.right(), rect.bottom()])

            person = api.identify_facecode(facecode)

            for track_id, face in shared['tracked_faces'].items():
                if FaceTracker.is_same_face(face['coords'], face_coordinates, max_rel_shift):
                    face["name"]=copy.copy(person['name'])
                    break

    @staticmethod
    def _verify_identify_local(shared, max_rel_shift):
        session = assert_session()
        persons = detect_and_identify_faces(shared['frame'], session)
        for person, rect, shapes in persons:
            face_coordinates = np.asarray(
                [rect.left(), rect.top(), rect.right(), rect.bottom()])

            for track_id, face in shared['tracked_faces'].items():
                if FaceTracker.is_same_face(face['coords'], face_coordinates, max_rel_shift):
                    face["name"]=copy.copy(person.name)
                    break
        session.close_all()

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
                _shared_data = self.memory_manager.dict()
                this_face = TrackedFace(_shared_data)
                track_id = this_face.id()
                self._shared['tracked_faces'][track_id]=_shared_data
                this_face.update_in_frame(face_coordinates)
                any_new_faces = True
            else:
                this_face.update_in_frame(face_coordinates)
            faces_in_frame.append((track_id, this_face))

        faces_recently_in_frame = []
        for track_id, face in self.tracked_faces.items():
            face.update_not_in_frame()
            if face._not_in_frame < self.missing_tol_nframes:
                faces_recently_in_frame.append((track_id, face))

        self.tracked_faces.clear()
        self._shared['tracked_faces'].clear()
        for track_id, face in faces_in_frame + faces_recently_in_frame:
            self.tracked_faces[track_id] = face
            self._shared['tracked_faces'][track_id] = face._shared

        self._shared['frame'] = frame.copy()

        # if any_new_faces:
        #     if self._identifier is None:

        return self.tracked_faces.values()

    def get_tracked_faces(self):
        return self.tracked_faces.copy()


class TrackedFace():
    """ a face tracked in video stream """

    def __init__(self, shared_data):

        self._tracker_id = uuid.uuid4()
        self._coords_buffer = deque(maxlen=5)
        self._shared = shared_data
        self._shared['id'] = self._tracker_id

    def name(self, *args, **kwargs):
        return self._shared.get('name',*args, **kwargs)

    def get(self, key, *args, **kwargs):
        return self._shared.get(key, *args, **kwargs)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            self._shared[key] = value

    def id(self):
        return self._tracker_id

    def coords(self):
        coords = np.mean(np.asarray(self._coords_buffer), axis=0).round().astype(int)
        return coords

    def update_in_frame(self, coords):
        self._not_in_frame = 0
        self._coords_buffer.append(coords)
        self._shared['coords'] = self.coords()

    def update_not_in_frame(self):
        self._not_in_frame += 1


