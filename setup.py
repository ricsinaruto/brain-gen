import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Read requirements
reqs = [
    line.strip()
    for line in (HERE / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="brain_gen",
    python_requires=">=3.13, <3.14",
    install_requires=reqs,
    packages=find_packages(),
)
