Human Face Recognition
===============================
author: Tobias Schoch

Overview
--------

A python library to provide out of the box human face identification, based on dlib public models.


Change-Log
----------
##### 0.0.3
* add facerec server client
* store absolute path in configfile
* fixed bug in db config path
* fixed path issues
* added persistent database path configuration
* add logging for existing database
* remove opencv dependency
* bug that a subsequent teach updates a previously unkown face
* fixed bug identify with no teached face crashes
* added models to manifest

##### 0.0.2
* add test data
* add the test data files
* fixed the threading issue with sqlalchemy etc
* proposal scoped session
* mad scoped session sqlalchemy
* add facetracker
* added requirements
* tests for dlib_api
* some work on dlib_api
* update readme
* facedb concepts ready

##### 0.0.1
* initial version


Installation / Usage
--------------------

To install use pip:

    pip install https://github.com/toschoch/python-facerec.git


Or clone the repo:

    git clone https://github.com/toschoch/python-facerec.git
    python setup.py install
    
Contributing
------------

TBD

Example
-------

TBD