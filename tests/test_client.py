from facerec.client import FacerecApi
from facerec.dlib_api import detect_faces
import pytest
import cv2
import os


here = os.path.split(__file__)[0]

@pytest.fixture()
def client():
    return FacerecApi("http://localhost:80")

def test_faces(client):
    print(client.faces())

def test_config(client):
    cfg = client.config()
    assert cfg['threshold']==0.6
    cfg = client.set_config(threshold=0.75)
    assert cfg['threshold']==0.75
    cfg = client.set_config(threshold=0.6)
    assert cfg['threshold']==0.6

def test_identify_image(client):
    img = cv2.imread(os.path.join(here, 'data', "Tobias_Schoch_TOS_big (Large).jpg"))
    print(client.identify_image(img))

def test_identify_code(client):
    img = cv2.imread(os.path.join(here, 'data', "Tobias_Schoch_TOS_big (Large).jpg"))
    facecode, rect, shape = detect_faces(img)[0]
    p = client.identify_facecode(facecode)
    print(p)

def test_modify_face_name(client):
    old = client.face(1)
    client.set_name(1,'Mickey Mouse')
    assert client.face(1)['name']=='Mickey Mouse'
    client.set_name(1,old['name'])
    assert client.face(1)['name']==old['name']

def test_delete_face(client):
    client.delete_face(2)
    print(client.faces())

def test_teach_code(client):
    img = cv2.imread(os.path.join(here, 'data', "Tobias_Schoch_TOS_big (Large).jpg"))
    facecode, rect, shape = detect_faces(img)[0]
    print(client.teach_facecode(facecode, name='Tobias Schoch'))

def test_teach_image(client):
    img = cv2.imread(os.path.join(here, 'data', "Tobias_Schoch_TOS_big (Large).jpg"))
    print(client.teach_image(img, name='Tobias Schoch'))