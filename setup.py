#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dfpipe",
    version="0.0.1",
    author="Leyia",
    author_email="x@leyia.fun",
    description="灵活、可扩展的DataFrame处理管道工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ciciy-l/dfpipe",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "dfpipe=dfpipe.cli:main",
        ],
    },
) 