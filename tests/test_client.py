from facerec.client import FacerecApi
from facerec.dlib_api import detect_faces
import pytest
import cv2
import os


here = os.path.split(__file__)[0]

@pytest.fixture()
def client():
    return FacerecApi("http://localhost:8081")

def test_faces(client):
    print(client.faces())

def test_names(client):
    print(client.names())

def test_identify_image(client):
    img = cv2.imread(os.path.join(here, 'data', "Tobias_Schoch_TOS_big (Large).jpg"))
    print(client.identify_image(img))

def test_identify_code(client):
    img = cv2.imread(os.path.join(here, 'data', "Tobias_Schoch_TOS_big (Large).jpg"))
    facecode, rect, shape = detect_faces(img)[0]
    p = client.identify_facecode(facecode)
    print(p)

def test_delete_face(client):
    client.delete_face(id=1)
    print(client.faces())

def test_teach_code(client):
    img = cv2.imread(os.path.join(here, 'data', "Tobias_Schoch_TOS_big (Large).jpg"))
    facecode, rect, shape = detect_faces(img)[0]
    print(client.teach_facecode(facecode, name='Tobias Schoch'))

def test_teach_image(client):
    img = cv2.imread(os.path.join(here, 'data', "Tobias_Schoch_TOS_big (Large).jpg"))
    print(client.teach_image(img, name='Tobias Schoch'))