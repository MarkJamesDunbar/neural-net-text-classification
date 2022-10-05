from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Kannada MNIST CNN model'
LONG_DESCRIPTION = 'Pytorch Neural Network aimed to classify letters from the Kannada MNIST dataset'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="kannadamnist_module",
    version=VERSION,
    author="Mark Dunbar",
    author_email="marjamdun@hotmail.co.uk",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['python', 'pytorch', 'mnist'],
    classifiers=[
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
