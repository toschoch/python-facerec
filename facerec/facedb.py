#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 20.04.2018
# author:  TOS

import logging
import pkg_resources
import pathlib

from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint, ARRAY, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, make_transient
from sqlalchemy import create_engine
from sqlalchemy import types, asc
from sqlalchemy.orm.exc import NoResultFound

import numpy as np


log = logging.getLogger(__name__)

__db_path = pathlib.Path(pkg_resources.resource_filename('facerec','data'))
__db_file = 'face.db'

__engine = None
nocommit = False

session = None

Base = declarative_base()

class FaceCode(types.TypeDecorator):

    impl = types.String

    def process_bind_param(self, value, dialect):
        return np.asarray(value).tobytes()[:self.impl.length]

    def process_result_value(self, value, dialect):
        return np.frombuffer(value)

    def copy(self, **kw):
        return FaceCode(self.impl.length)


class Person(Base):
    __tablename__ = 'persons'
    # Here we define columns for the table person
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    name = Column(String(250))
    code = Column(FaceCode(1024), nullable=False)
    nmeans = Column(Float, nullable=False, default=1.0)

    #code = Column(ARRAY(Float, dimensions=1024))


class Image(Base):
    __tablename__ = 'images'
    # Here we define columns for the table person
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, ForeignKey("persons.id"), primary_key=True)
    filename = Column(String(250), nullable=False, unique=True)
    person_id = Column(Integer, ForeignKey("persons.id"))


def get_db_path():
    return __db_path

def get_db_file():
    return __db_path.joinpath(__db_file)

def set_db_path(path):

    global __db_path

    path = pathlib.Path(path)
    assert path.exists()
    __db_path = path
    open_db()

def open_db():

    global __engine
    global session

    __engine = create_engine('sqlite:///{}'.format(get_db_file()))
    Base.metadata.create_all(__engine)
    session = Session(bind=__engine)

def assert_db_open():
    if session is None:
        open_db()

def persons():
    assert_db_open()
    return session.query(Person).order_by(asc(Person.id)).all()

def images():
    assert_db_open()
    return session.query(Image).all()

def find_similar_persons(encoding):
    """
    returns the sql persons with similar faces corresponding to the encoding, sorted by similarity.
    Args:
        encoding: (np.array, cv.array) with 128-d face feature vector.

    Returns:
        (list of Persons) with
    """
    #TODO: remove loading of all encodings
    encodings = np.vstack((p.code for p in persons()))
    distances = np.sqrt(np.sum((encodings - encoding[None,:])**2,axis=1))
    #I = np.argsort(distances)
    I = np.where(distances < 0.7)[0]
    similar_persons = np.asarray(session.query(Person).filter(Person.id.in_((I+1).tolist())).all())
    similar_persons = similar_persons[np.argsort(distances[I])].tolist()
    return similar_persons


def get_person(name=None, id=None):
    """
    returns the first sql person corresponding to the name
    Args:
        name: (str) name of the person in the face database (either name or id has to be specified)
        id: (int) id of the person in the face database (either name or id has to be specified)

    Returns:
        (facedb.Person)

    """
    if id is not None:
        return session.query(Person).filter(Person.id == id).one()
    elif name is not None:
        return session.query(Person).filter(Person.name == name).one()
    else:
        raise ValueError("either name or id must be specified!")

def teach(facecode, name=None, id=None, weight=1.0):
    """
    teach the classifier that the facecode is of the specified person
    Args:
        facecode:
        name: (str) name of the person in the face database (either name or id has to be specified)
        id: (int) id of the person in the face database (either name or id has to be specified)
        weight: (float, default 1.0) weight to assign to specific face code in teaching.

    Returns:
        Person
    """

    facecode = np.asarray(facecode)
    assert_db_open()

    try:
        p = get_person(name, id)

        # update the centroid
        new_nmeans = p.nmeans + weight
        p.code = ((p.code * p.nmeans) + facecode * weight) / new_nmeans
        p.nmeans = new_nmeans

    except NoResultFound:
        if name is None:
            raise ValueError("face unknown and no name is specified! Specify a name...")
        p = Person(name=name, code=facecode)
        session.add(p)

    if not nocommit:
        session.commit()

    return p

def identify_person(facecode):
    """
    search for similar faces in database, return most similar person and assure the entry for every unknown face.
    Args:
        facecode: (np.array, cv.array) with 128-d face feature vector.

    Returns:
        (facedb.Person)
    """

    facecode = np.asarray(facecode)
    assert_db_open()

    similar_persons = find_similar_persons(facecode)

    if len(similar_persons) == 0:
        log.info("found unknown face add to database...")
        p = Person(name='unknown', code=facecode)
        session.add(p)

    elif len(similar_persons) == 1:
        p = similar_persons[0]

    else:
        log.info("found multiple candidates but return the most similar...")
        p = similar_persons[0]

    if not nocommit:
        session.commit()

    return p
