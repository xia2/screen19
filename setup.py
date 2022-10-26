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
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "screen19 = screen.screen:main"
            "screen19.minimum_exposure = screen.minimum_exposure:main"
        ],
        "libtbx.dispatcher.script": [
            "screen19.minimum_exposure = screen19.minimum_exposure"
        ],
        "libtbx.precommit": ["screen19 = screen19"],
    },
    install_requires=['python_version>="3.8"'],
    license="BSD license",
    long_description="\n\n".join([readme, changelog_header, changelog]),
    include_package_data=True,
    name="screen19",
    packages=find_packages(where="src/screen"),
    test_suite="tests",
    tests_require=["pytest"],
    version="0.213",
    zip_safe=False,
)
