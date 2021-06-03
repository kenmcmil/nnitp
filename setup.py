import codecs
import os
import platform

from setuptools import setup, find_packages


# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
try:
  with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
      long_description = f.read()
except:
  # This happens when running tests
  long_description = None

setup(name='nnitp',
      version='0.1',
      description='Neural net interpretability with Baysian interpolants',
      long_description=long_description,
      url='https://github.com/microsoft/nnitp',
      author='nnitp team',
      author_email='nomail@example.com',
      license='MIT',
#      packages=find_packages(),
      packages = ['nnitp','nnitp.models'],
      package_data={'nnitp':['models/*.h5',]},
      install_requires=[
          'wheel',
          'traitsui',
          'numpy',
          'matplotlib',
          'torch',
          'torchvision',
          'PyQt5'
      ],
      entry_points = {
        'console_scripts': ['nnitp=nnitp.nnitp:main',],
      },
      zip_safe=False)
