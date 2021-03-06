from setuptools import setup, find_packages
from pathlib import Path

requirements = [
    r
    for r in Path("requirements.txt").read_text().splitlines()
    if '@' not in r
]

setup(
    name='TSAug',
    version='0.1.0',    
    description='A timeseries augmentation library',
    url='https://github.com/Takezo87/tsaug',
    author='Johannes Nowak',
    author_email='nowakj@gmx.de',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        ]
)
