from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

conda_env_file = "environment.yml"
readme_file = "README.md"
pip_req_file = "requirements.txt"

def read(fname):
    with open(path.join(here,fname),"rb", "utf-8") as fp:
        content = fp.read()
    return content
# Get the long description from the README file
long_description = read(readme_file)

# get the dependencies and installs
all_reqs = []
all_reqs += read(pip_req_file).splitlines()

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip() for x in all_reqs if x.startswith('git+')]

setup(
    name='python-facerec',
    version_format='{tag}.dev{commitcount}',
    setup_requires=['setuptools-git-version','pytest-runner'],
    description='A python library to provide out of the box human face identification, based on dlib public models.',
    long_description=long_description,
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    entry_points={'console_scripts': []},
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author='Tobias Schoch',
    install_requires=install_requires,
    tests_require=['pytest','opencv-python>=3.4.0'],
    dependency_links=dependency_links,
    author_email='tobias.schoch@helbling.ch'
)
