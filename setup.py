#!/usr/bin/env python

from setuptools import setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

changelog_header = """
*********
Changelog
*********
"""

with open("CHANGELOG.rst") as changelog_file:
    changelog = changelog_file.read()

setup()
