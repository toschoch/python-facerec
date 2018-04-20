from facerec import facedb
from facerec.facedb import Person, Image
import sqlalchemy.orm
import os
import numpy as np
import pytest

# Sample Test passing with nose and pytest
def test_path():
    print(facedb.get_db_path())

def test_db_creation():
    facedb.open_db()
    assert os.path.exists(facedb.get_db_file())
    assert os.path.isfile(facedb.get_db_file())

def test_persons():
    print(facedb.persons())
    code = np.random.rand(128)
    facedb.session.add(facedb.Person(name='Tobias Schoch',code=code))

    p = facedb.persons()[0]
    assert np.all(p.code==code)


def test_comparison():
    facedb.nocommit = True
    facedb.assert_db_open()
    codes = np.random.rand(10,128)
    for i in range(10):
        code = codes[i,:]
        facedb.teach(code, name='Tobias Schoch {}'.format(i))
        # facedb.session.add(facedb.Person(name='Tobias Schoch {}'.format(i),code=code))

    p = facedb.persons()[3]

    # teach again
    facedb.teach(codes[6,:]+np.random.randn(128)*0.01, "Tobias Schoch 6", 0.8)
    assert (facedb.find_similar_persons(codes[6,:])[0].name) == "Tobias Schoch 6"

    # test comparison
    assert np.all(p.code==codes[3,:])

    assert np.all(facedb.find_similar_persons(codes[4,:])[0].code == codes[4,:])

    assert np.all(facedb.get_person("Tobias Schoch 5").code == codes[5,:])

    with pytest.raises(sqlalchemy.orm.exc.NoResultFound):
        facedb.get_person('Mickey Mouse')
