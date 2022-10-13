#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='dlnpyutils',
      version='1.0.47',
      description='David Nidever Python Utility Functions',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/dlnpyutils',
      packages=['dlnpyutils'],
      scripts=['bin/job_daemon','bin/bump','bin/topypi'],
      install_requires=['numpy','astropy','scipy']
)
