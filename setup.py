#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


with open('README.rst') as readme_file:
    readme = readme_file.read()


requirements = [s.strip() for s in open('requirements.txt', 'r').readlines()]
test_requirements = []


setup(
    name='lm_kbc',
    version='0.1.0',
    description="Research project for...",
    long_description=readme,
    author="Cristian Berrio Aroca",
    author_email='cberrio@expert.ai',
    url='https://github.com/expertailab/Textual-Entailment-for-Effective-Triple-Validation-in-Object-Prediction',
    python_requires='>=3.5',
    packages=find_packages(include=["lm_kbc", "lm_kbc.*"]),
    package_dir={'lm_kbc':
                 'lm_kbc'},
    include_package_data=True,
    install_requires=requirements,
    license="",
    zip_safe=False,
    keywords='lm_kbc',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
