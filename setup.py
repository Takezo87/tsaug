from setuptools import setup
from pathlib import Path

requirements = [
    r
    for r in Path("requirements.txt").read_text().splitlines()
    if '@' not in r
]

setup(
    name='tsaug',
    version='0.1.0',    
    description='A timeseries augmentation library',
    url='https://github.com/Takezo87/tsaug',
    author='Johannes Nowak',
    author_email='nowakj@gmx.de',
    license='BSD 2-clause',
    packages=['tsaug'],
    install_requires=requirements,

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
