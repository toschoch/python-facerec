#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 23.04.2018
# author:  TOS

import logging
import uuid
import numpy as np
import time
import copy
from collections import deque
from multiprocessing import Manager, Event # Process as Thread
from threading import Thread #as Process, Event

from .dlib_api import detect_and_identify_faces, detect_faces
from .facedb import assert_session, unknown_tag
from .client import FacerecApi

import dlib

log = logging.getLogger(__name__)

_processes = []

class Identifier(Thread):
    def __init__(self, interval, function, args=[], kwargs={}):
        super(Identifier, self).__init__()
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.canceled = Event()
        self.triggered = Event()

    def cancel(self):
        """Stop the timer if it hasn't finished yet"""
        self.canceled.set()
        self.triggered.clear()

    def trigger(self):
        self.triggered.set()
        self.triggered.clear()

    def run(self):
        while not self.canceled.is_set():
            self.function(*self.args, **self.kwargs)
            self.triggered.wait(self.interval)



class FaceTracker(object):
    """ Trackes Faces in an concurent stream of images. """
    def __init__(self, url=None, max_relative_shift=0.8, avg_over_nframes=1, missing_tolerance_nframes=0,
                 identification_interval=2, appearance_callback=None, identification_callback=None, disappearance_callback=None):
        """
        creates a face tracker that identifies the tracked face with facerec engine. The requests to identify can be done locally
        or sent to a facerec server by specifying the url.
        Args:
            url: (optional) url of the facerec server
            max_relative_shift: (float) maximal shift of face from one frame to the other relative to face width (to be tracked as the same face)
            avg_over_nframes:  (int) optional moving average filter of the face position
            missing_tolerance_nframes: (int) how many frames should the tracker store a face that is not detected anymore (helps for short missing detection)
            identification_interval: (float) seconds to repeat the identification request
            appearance_callback: (face->None) function to call on appeared faces
            identification_callback: (face->None) function to call on completed identification
            disappearance_callback: (face->None) function to call on disappeared faces
        """

        self.max_rel_shift = max_relative_shift
        self.avg_over_nframes = avg_over_nframes
        self.missing_tol_nframes = missing_tolerance_nframes

        self.detector = dlib.get_frontal_face_detector()

        self.memory_manager = Manager()
        self._shared = self.memory_manager.dict()
        self._shared['tracked_faces'] = self.memory_manager.dict()
        self.tracked_faces = {}

        self.appearance_callback = appearance_callback
        self.disappearance_callback = disappearance_callback

        if url is not None:
            self.api = FacerecApi(url)
            self._identifier = Identifier(identification_interval, FaceTracker._verify_identify_server,
                                          args=(self._shared, self.api, self.max_rel_shift,
                                                identification_callback, disappearance_callback))
        else:
            self.api = None
            self._identifier = Identifier(identification_interval, FaceTracker._verify_identify_local,
                                          args=(self._shared, self.max_rel_shift,
                                                identification_callback, disappearance_callback))

        self._identifier.start()

    @staticmethod
    def _verify_identity(shared, max_rel_shift, rect, name, id, identification_callback, disappearance_callback):
            face_coordinates = np.asarray(
                [rect.left(), rect.top(), rect.right(), rect.bottom()])

            for track_id, face in shared['tracked_faces'].items():
                if FaceTracker.is_same_face(face['coords'], face_coordinates, max_rel_shift):
                    name=copy.copy(name)
                    id=copy.copy(id)
                    if face["face_id"] != id:
                        if face["identified"]:
                            FaceTracker._on_disappearance_proxy(face, disappearance_callback)
                        face["name"] = name
                        face["face_id"] = id
                        face['identified'] = time.time()
                        log.info("identified: {}".format(face))
                        if identification_callback is not None:
                            identification_callback(face)
                        break

    @staticmethod
    def _verify_identify_server(shared, api, max_rel_shift, identification_callback, disappearance_callback):
        if 'frame' not in shared: return
        faces = detect_faces(shared['frame'])
        for facecode, rect, shapes in faces:

            person = api.identify_facecode(facecode)
            FaceTracker._verify_identity(shared, max_rel_shift, rect,
                                         copy.copy(person['name']), copy.copy(person['id']),
                                         identification_callback, disappearance_callback)

    @staticmethod
    def _verify_identify_local(shared, max_rel_shift, identification_callback, disappearance_callback):
        if 'frame' not in shared: return
        session = assert_session()
        persons = detect_and_identify_faces(shared['frame'], session)
        for person, rect, shapes in persons:
            FaceTracker._verify_identity(shared, max_rel_shift, rect,
                                         copy.copy(person.name), copy.copy(person.id),
                                         identification_callback, disappearance_callback)
        session.close_all()

    @staticmethod
    def start_thread(shared, callback, proxy):
        p = Thread(target=proxy, args=(shared, callback))
        _processes.append(p)
        p.start()

    @staticmethod
    def _on_appearance_proxy(face, callback):
        face['appeared'] = time.time()
        log.info("appeared: {}".format(face))
        if callback is not None:
            callback(face)

    @staticmethod
    def _on_disappearance_proxy(face, callback):
        if not face['disappeared']:
            face['disappeared'] = time.time()
        log.info("disappeared: {}".format(face))
        if callback is not None:
            callback(face)

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
                this_face = TrackedFace(self.memory_manager.dict())
                FaceTracker.start_thread(this_face._shared, self.appearance_callback, FaceTracker._on_appearance_proxy)

                track_id = this_face.id()
                self._shared['tracked_faces'][track_id]=this_face._shared

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
                face._shared['disappeared'] = time.time()

        tracked_faces = dict(faces_in_frame + faces_recently_in_frame)
        for track_id, face in self.tracked_faces.items():
            if track_id not in tracked_faces.keys():
                FaceTracker.start_thread(face._shared, self.disappearance_callback, FaceTracker._on_disappearance_proxy)

        self.tracked_faces.clear()
        self._shared['tracked_faces'].clear()
        for track_id, face in tracked_faces.items():
            self.tracked_faces[track_id] = face
            self._shared['tracked_faces'][track_id] = face._shared

        self._shared['frame'] = frame.copy()

        if any_new_faces:
            self._identifier.trigger()

        return self.tracked_faces.values()

    def get_tracked_faces(self):
        return self.tracked_faces.copy()

    def stop(self):
        [p.join() for  p in _processes]
        self._identifier.cancel()
        self._identifier.join()

class TrackedFace():
    """ a face tracked in video stream """

    def __init__(self, shared_data):

        self._processes = []
        self._tracker_id = str(uuid.uuid4())
        self._coords_buffer = deque(maxlen=5)
        self._shared = shared_data
        self._shared['id'] = self._tracker_id
        self._shared['face_id'] = -1
        self._shared['name'] = unknown_tag
        self._shared['identified'] = False
        self._shared['disappeared'] = False


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
        self._shared['coords'] = self.coords().tolist()

    def update_not_in_frame(self):
        self._not_in_frame += 1
