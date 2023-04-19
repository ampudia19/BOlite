"""BOlite"""
from pathlib import Path
from setuptools import find_packages
from setuptools import setup


def read_lines(path):
    """Read lines of `path`."""
    with open(path) as f:
        return f.read().splitlines()

BASE_DIR = Path(__file__).parent

setup(
    name="BOlite",
    long_description=open(BASE_DIR / "README.md").read(),
    install_requires=read_lines(BASE_DIR / "requirements.txt"),
    packages=find_packages(exclude=["docs"]),
    version="0.1.0",
    description="Bayesian Optimisation library for instructive purposes",
    author="Innovation Growth Lab",
    license="MIT",
)
