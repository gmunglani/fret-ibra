from setuptools import setup
import os
import ibra

# Fix so that the setup.py usage is CWD-independent
SETUPDIR = os.path.abspath(os.path.dirname(__file__))
reqsfname = os.path.join(SETUPDIR, 'requirements.txt')
reqs = open(reqsfname, 'r').read().strip().splitlines()

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

setup(
    name = "fret-ibra",
    packages = ["ibra"],
    install_requires=reqs,
    version = "0.3.2",
    license='BSD',
    description = "FRET-IBRA is used to process fluorescence resonance energy transfer (FRET) intensity data to produce ratiometric images for further analysis",
    long_description = long_descr,
    author = "Gautam Munglani",
    author_email = "gmunglani@gmail.com",
    url = "https://github.com/gmunglani/fret-ibra",
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': ['ibra=ibra.__main__:main'],
    },
    python_requires='>=3.5',
    keywords=['opencv', 'background subtraction', 'DBSCAN clustering', 'FRET imaging', 'ratiometric'],
    zip_safe=True,
)

