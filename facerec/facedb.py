#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 20.04.2018
# author:  TOS

import logging
import pkg_resources
import pathlib
import os

from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint, ARRAY, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, Session as SqlSession, make_transient_to_detached, make_transient
from sqlalchemy import create_engine
from sqlalchemy import types, asc
from sqlalchemy.orm.exc import NoResultFound
import json

import numpy as np

log = logging.getLogger(__name__)

unknown_tag = "unknown"

__db_config_file = pathlib.Path(os.path.expanduser('~/.facerec.json'))
__distance_threshold = 0.6
try:
    with open(__db_config_file,'r') as fp:
        __db_path = pathlib.Path(json.load(fp)['path'])
    with open(__db_config_file, 'r') as fp:
        __distance_threshold = float(json.load(fp).get('threshold',__distance_threshold))
except:
    __db_path = pathlib.Path(pkg_resources.resource_filename('facerec','data'))
__db_file = 'face.db'

__engine = None
nocommit = False
Session = None

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


def get_db_path():
    return __db_path

def get_db_file():
    return __db_path.joinpath(__db_file)


def _update_config_file(new_config):
    log.info("write config file {}...".format(__db_config_file))
    config = {}
    __db_config_file_exp = pathlib.Path(os.path.expanduser(__db_config_file))
    if __db_config_file_exp.exists():
        with open(__db_config_file_exp, 'r') as fp:
            config = json.load(fp)
    else:
        config = {}
    config.update(new_config)
    with open(__db_config_file_exp, 'w+') as fp:
        json.dump(config, fp)

def set_db_path(path, persistent=False):

    global __db_path

    path = pathlib.Path(path)
    assert path.exists()
    if persistent:
        _update_config_file({'path': str(path)})
    __db_path = path
    open_db()

def get_distance_threshold():
    return __distance_threshold

def set_distance_threshold(threshold, persistent=False):

    global __distance_threshold

    __distance_threshold = threshold
    if persistent:
        _update_config_file({'threshold':threshold})

def open_db():

    global __engine
    global Session

    __engine = create_engine('sqlite:///{}'.format(get_db_file()))
    if get_db_file().exists():
        log.info("Use existing database ({})...".format(get_db_file()))
    else:
        log.info("Create new empty database ({})...".format(get_db_file()))
    Base.metadata.create_all(__engine)
    session_factory = sessionmaker(bind=__engine)
    Session = scoped_session(session_factory)

def assert_session(session=None):
    if Session is None:
        open_db()
    if session is None:
        session = Session()
    assert isinstance(session, SqlSession)
    return session


def persons(session=None):
    session = assert_session(session)
    return session.query(Person).order_by(asc(Person.id)).all()

def find_similar_persons(encoding, session=None):
    """
    returns the sql persons with similar faces corresponding to the encoding, sorted by similarity.
    Args:
        encoding: (np.array, cv.array) with 128-d face feature vector.
        session: (sqlalchemy.Session, optional) session to use for db lookup

    Returns:
        (list of Persons) with
    """
    session = assert_session(session)

    #TODO: remove loading of all encodings
    _persons = [p.code for p in persons()]
    if len(_persons) < 1:
        return []
    encodings = np.vstack(_persons)
    distances = np.sqrt(np.sum((encodings - encoding[None,:])**2,axis=1))
    #I = np.argsort(distances)
    I = np.where(distances < __distance_threshold)[0]
    similar_persons = np.asarray(session.query(Person).filter(Person.id.in_((I+1).tolist())).all())
    similar_persons = similar_persons[np.argsort(distances[I])].tolist()
    return similar_persons


def get_person(name=None, id=None, session=None):
    """
    returns the first sql person corresponding to the name
    Args:
        name: (str) name of the person in the face database (either name or id has to be specified)
        id: (int) id of the person in the face database (either name or id has to be specified)
        session: (sqlalchemy.Session, optional)

    Returns:
        (facedb.Person)

    """

    session = assert_session(session)

    if id is not None:
        return session.query(Person).filter(Person.id == id).one()
    elif name is not None:
        return session.query(Person).filter(Person.name == name).one()
    else:
        raise ValueError("either name or id must be specified!")

def teach(facecode, name=None, id=None, weight=1.0, session=None):
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
    session = assert_session(session)

    try:
        p = get_person(name, id)

    except NoResultFound:

        if name is None:
            raise ValueError("face unknown and no name is specified! Specify a name...")

        # find similar face and update name
        p = identify_person(facecode, session)
        p.name = name

    # teach
    # update the centroid
    new_nmeans = p.nmeans + weight
    p.code = ((p.code * p.nmeans) + facecode * weight) / new_nmeans
    p.nmeans = new_nmeans

    if not nocommit:
        session.commit()

    return p

def identify_person(facecode, session=None):
    """
    search for similar faces in database, return most similar person and assure the entry for every unknown face.
    Args:
        facecode: (np.array, cv.array) with 128-d face feature vector.
        session: (sqlalchemy.Session, optional)

    Returns:
        (facedb.Person)
    """

    facecode = np.asarray(facecode)
    session = assert_session(session)

    similar_persons = find_similar_persons(facecode)

    if len(similar_persons) == 0:
        log.info("found unknown face add to database...")
        p = Person(name=unknown_tag, code=facecode)
        session.add(p)

    elif len(similar_persons) == 1:
        p = similar_persons[0]

    else:
        log.info("found multiple candidates but return the most similar...")
        p = similar_persons[0]

    if not nocommit:
        session.commit()

    return p

def close():
    if Session is not None:
        Session.remove()
