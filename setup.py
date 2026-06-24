#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'cdi',
    version          = '0.0.1',
    author           = 'The CDI Project Developers',
    license          = 'BSD-2-Clause',
    classifiers      = [
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: BSD License',
    ],
    description      = (
        'Conditional diffusion models for uncertainty-aware CBED inverse '
        'problems.'
    ),
    packages         = setuptools.find_packages(
        include = [ 'cdi', 'cdi.*' ]
    ),
    package_data     = {
        'cdi' : [
            'bundled/diffusion/LICENSE',
            'bundled/leanbase/LICENSE',
            'nn/conditional_dm/LICENSE',
            'nn/conditional_dm/NOTICE.md',
        ],
    },
    install_requires = [  ],
)
