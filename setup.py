"""Setup script for revisiting_rainbow.

This script will install the algorithms presented in revisiting_rainbow paper as a Python module.

See: https://github.com/JohanSamir/revisiting_rainbow
"""

import pathlib
from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

install_requires = [
    'dopamine-rl>= 3.1.10 ',
    ]

rev_rainbow_description = (
    'Revisiting Rainbow: Promoting more insightful and inclusive deep reinforcement learning research')

setup(
    name='revisiting_rainbow',
    version='1.0.0',
    description=rev_rainbow_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/JohanSamir/revisiting_rainbow',
    author='Johan S Obando-Ceron and Pablo Samuel Castro',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

    ],
    keywords='dopamine, reinforcement, machine, learning, research',
    install_requires=install_requires,
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/JohanSamir/revisiting_rainbow/issues',
        'Source': 'https://github.com/JohanSamir/revisiting_rainbow',
    },
    license='Apache 2.0',
)

