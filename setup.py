##############################################################################
#
# Setup file. Setup is not really needed, anyway you can consider it as a
# solution in case of execution issues.
#
##############################################################################
from setuptools import setup, find_packages
import os
import webbrowser

__version__ = '0.7.6'
NAME = 'SNN_GDB'
AUTHORS = 'Ruggero Basanisi, ' \
          'Andrea Brovelli, ' \
          'Emilio Cartoni, ' \
          'Gianluca Baldassarre'
MAINTEINER = 'Ruggero Basanisi'
PAPER_PREPRINT = 'https://doi.org/10.1101/867366'
GITHUB_URL = \
    'https://github.com/GOAL-Robots/ASpikingModelOfGoalDirectedBehaviour'
EXTRA_REQ = {'hdf5 files r/w': ['h5py']}

setup(
    name=NAME,
    version=__version__,
    packages=find_packages(),
    author=AUTHORS,
    maintainer=MAINTEINER,
    description="A spiking neural network for goal-directed behaviour",
    long_description=webbrowser.open(PAPER_PREPRINT),
    url=GITHUB_URL,
    license='MIT License',
    install_requires=['numpy',
                      'scipy',
                      'matplotlib'],
    extra_requires=EXTRA_REQ
)
