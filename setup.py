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
        "console_scripts": [
            "i19.screen = screen19.screen:main",
            "screen19 = screen19.screen:main",
            "i19.stability_fft = screen19.stability_fft:main",
            "i19.minimum_exposure = screen19.minimum_exposure:main",
            "screen19.minimum_exposure = screen19.minimum_exposure:main",
        ],
        "libtbx.dispatcher.script": [
            "i19.screen = i19.screen",
            "screen19 = screen19",
            "i19.stability_fft = i19.stability_fft",
            "i19.minimum_exposure = i19.minimum_exposure",
            "screen19.minimum_exposure = screen19.minimum_exposure",
        ],
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
