#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'evlearn',
    version          = '0.0.1',
    author           = 'Dmitrii Torbunov',
    author_email     = 'dtorbunov@bnl.gov',
    classifiers      = [
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = "evlearn",
    packages         = setuptools.find_packages(
        include = [ 'evlearn', 'evlearn.*' ]
    ),
    install_requires = [
	"h5py >= 3.0.0",
	"numpy >= 1.20.0",
	"pandas >= 2.0.0",
	"scipy >= 1.10.0",
	"tqdm >= 4.0.0",
	"torch >= 2.2.0",
	"torchvision >= 0.17.0",
	"psee_adt",
    ],
)

