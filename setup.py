#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

changelog_header = """
*********
Changelog
*********
"""

with open("CHANGELOG.rst") as changelog_file:
    changelog = changelog_file.read()

setup(
    author="Diamond Light Source",
    author_email="scientificsoftware@diamond.ac.uk",
    description="Screening program for small-molecule single-crystal X-ray diffraction "
    "data",
    entry_points={
        "console_scripts": [],
        "libtbx.dispatcher.script": [],
        "libtbx.precommit": ["screen19 = screen19"],
    },
    install_requires=['typing;python_version<"3.5"', "procrunner"],
    license="BSD license",
    long_description="\n\n".join([readme, changelog_header, changelog]),
    include_package_data=True,
    name="screen19",
    packages=find_packages(),
    test_suite="tests",
    tests_require=["mock>=2.0", "pytest>=4.5"],
    version="0.213",
    zip_safe=False,
)
