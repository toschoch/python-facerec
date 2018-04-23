try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution('facerec').version
except (pkg_resources.DistributionNotFound, ImportError):
    __version__ = 'dev'

from .dlib_api import detect_and_identify_faces, teach_person