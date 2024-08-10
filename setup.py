from setuptools import find_packages, setup

with open("./requirements.txt") as text_file:
    requirements = text_file.readlines()

requirements = list(map(lambda x: x.rstrip("\n"), requirements))

setup(
    name="auto_matcher",
    version="0.1",
    description="A sample Python package for Auto Matching Source and Target Schema",
    author="SHUBHAM GUPTA",
    author_email="2022AA05062@wilp.bits-pilani.ac.in",
    packages=find_packages(include=["auto_matcher", "auto_matcher.*"]),
    install_requires=requirements,
)
