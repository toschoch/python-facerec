#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 30.04.2018
# author:  TOS

import requests
import base64
import json
from io import BytesIO
import numpy as np

class FacerecApi(object):

    def __init__(self, url='http://localhost:80'):
        self.url = url

    def faces(self):
        return requests.get(self.url+'/faces').json()

    def config(self):
        return requests.get(self.url+'/config').json()

    def set_config(self, **kwargs):
        for parameter, new_value in kwargs.items():
            requests.patch(self.url+'/config/{}'.format(parameter), data={'value': new_value}).json()
        return self.config()

    def face(self, id_or_name):
        return requests.get(self.url+'/faces/{}'.format(id_or_name)).json()

    def delete_face(self, id_or_name):
        return requests.delete(self.url+'/faces/{}'.format(id_or_name)).json()

    def set_name(self, id_or_name, new_name):
        return requests.patch(self.url+'/faces/{}'.format(id_or_name), data={'name':new_name}).json()

    def identify_facecode(self, code):
        # encode face code and send to identify
        payload = {'code': base64.b64encode(np.asarray(code).tobytes()).decode('ascii')}
        r = requests.post(self.url+'/facecode/identify', json=payload)
        return r.json()

    def teach_facecode(self, code, name=None, id=None):
        # encode face code and send to identify
        payload = {'code': base64.b64encode(np.asarray(code).tobytes()).decode('ascii')}
        if name is not None:
            payload['name'] = name
        if id is not None:
            payload['id'] = id
        r = requests.post(self.url+'/facecode/identify', json=payload)
        return r.json()

    def identify_image(self, image):
        buffer = self.compress_image(image)
        # encode face code and send to identify
        payload = {'image': base64.b64encode(buffer).decode('ascii')}
        r = requests.post(self.url+'/image/teach', json=payload)
        return r.json()

    def teach_image(self, image, name=None, id=None):
        # encode face code and send to identify
        buffer = self.compress_image(image)
        payload = {'image': base64.b64encode(buffer).decode('ascii')}
        if name is not None:
            payload['name'] = name
        if id is not None:
            payload['id'] = id
        r = requests.post(self.url+'/image/teach', json=payload)
        return r.json()


    @staticmethod
    def compress_image(image):
        try:
            import cv2
            _, array = cv2.imencode('.jpg',image)
            return array.tostring()
        except ModuleNotFoundError:
            from PIL import Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert('RGB')
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            return buffer.getvalue()
