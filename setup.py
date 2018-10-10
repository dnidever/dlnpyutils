#!/usr/bin/env python

from distutils.core import setup

setup(name='dlnpyutils',
      version='1.0',
      description='David Nidever Python Utility Functions',
      author='David Nidever',
      author_email='dnidever@noao.edu',
      url='https://github.com/dnidever/dlnpyutils',
      packages=['dlnpyutils'],
      requires=['numpy','astropy','scipy']
)
