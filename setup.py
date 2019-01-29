from setuptools import setup
import ibra

reqsfname = os.path.join(SETUPDIR, 'requirements.txt')
reqs = open(reqsfname, 'r', encoding='utf-8').read().strip().splitlines()

def readme():
    with open('README.md') as f:
        return f.read().decode("utf-8")

setup(
    name = "fret-ibra",
    packages = ["ibra"],
    install_requires=reqs,
    version = ibra.__version__,
    license='BSD',
    description = "FRET-IBRA is used to process fluorescence resonance energy transfer (FRET) intensity data to produce ratiometric images for further analysis",
    long_description = readme(),
    author = "Gautam Munglani",
    author_email = "gmunglani@gmail.com",
    url = "https://github.com/gmunglani/fret-ibra.git",
    entry_points={
        'console_scripts': ['ibra = ibra:main'],
    },
    python_requires='2.7',
    keywords=['opencv', 'background subtraction', 'DBSCAN clustering', 'FRET imaging', 'ratiometric'],
    zip_safe=True,
)

