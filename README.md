Human Face Recognition
===============================
author: Tobias Schoch

Just to make it clear from the beginning. This is just a convenient wrapper of the awsome _dlib_
library (<https://pypi.org/project/dlib/>) by Davis King. **He did the hard work!!**

Overview
--------

A python library to provide out of the box human face identification, based on dlib public models.


Change-Log
----------
##### 0.1.3
* improved logging events of tracker
* try again threads
* log info for identified face
* use secrets in dron
* try multiprocessing instead of thread

##### 0.1.2
* added parameter for identification interval
* changed package name to facerec, fixed error in thread if no frame to process

##### 0.1.1
* removed pathlib expanduser as not available in python3.4
* fix python 3.3. setup issue
* switched to thread as only this works on my machine, removed buggy unneccesary changelog

##### 0.1.0
* made identification more rigorous and improved client
* added threshold to configuration
* removed recursion included trigger on new face
* fix uuid to string
* returned to threads for on_appearance/disappearance slots
* removed unwanted logging configuration
* added disapearance time
* added on appear on disappear slots
* added real multiprocessing for performant stream

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