from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A library to make quaternion neural networks.'
LONG_DESCRIPTION = "A python library which has quaternion versions of some of the mostly used libraries in torch.nn. It supports all of the features of the supported layers. Main advantage of quaternion layers is that, it has only 1/4th the number of weights compared to it's real counterpart"

# Setting up
setup(
    name="qytorch",
    version=VERSION,
    author="smlab-niser",
    author_email="smishra@niser.ac.in",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(include=["qytorch", "qytorch.*"]),
    install_requires=[
        'torch',
        'numpy',
        # 
    ],
    keywords=['quaternion', 'neural networks', 'torch', 'machine learning', 'parameter reduction'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    url="https://github.com/smlab-niser/qytorch",
    license="MIT",
)
