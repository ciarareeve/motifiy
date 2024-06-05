from setuptools import setup, find_packages

setup(
    name='motify',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'flask',
        'biopython',
        'scikit-learn',
        'weblogo',
        'logomaker'
    ],
    entry_points={
        'console_scripts': [
            'motify=main:main',
        ],
    },
)
