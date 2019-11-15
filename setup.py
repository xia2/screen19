#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = ["procrunner", "typing"]
test_requirements = ["mock>=2.0", "pytest>=4.5"]

setup(
    author="Diamond Light Source",
    author_email="scientificsoftware@diamond.ac.uk",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
    ],
    description="Screening program for small-molecule single-crystal X-ray diffraction "
    "data",
    entry_points={
        "console_scripts": [
            "i19.screen = screen19.screen:main",
            "screen19 = screen19.screen:main",
            "i19.stability_fft = screen19.stability_fft:main",
            "i19.sync = screen19.sync:main",
            "i19.minimum_exposure = screen19.minimum_exposure:main",
            "screen19.minimum_exposure = screen19.minimum_exposure:main",
        ],
        "libtbx.dispatcher.script": [
            "i19.screen = i19.screen",
            "screen19 = screen19",
            "i19.stability_fft = i19.stability_fft",
            "i19.sync = i19.sync",
            "i19.minimum_exposure = i19.minimum_exposure",
            "screen19.minimum_exposure = screen19.minimum_exposure",
        ],
        "libtbx.precommit": ["screen19 = screen19"],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    name="screen19",
    packages=find_packages(),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/xia2/screen19",
    version="0.206",
    zip_safe=False,
)
