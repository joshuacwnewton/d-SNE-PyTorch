# Stdlib imports
from setuptools import setup

setup(
    name='dsne_pytorch',
    version='0.1.0',
    packages=[
        'dsne_pytorch',
        'dsne_pytorch/data_loading',
        'dsne_pytorch/model'
    ],
    entry_points={
        'console_scripts': [
            'dsne_pytorch = dsne_pytorch.__main__:main'
        ]
    },
    install_requires=[
        "h5py",
        "numpy",
        "pandas",
        "opencv-python",
        "torch",
        "torchvision",
        "tensorboard"
    ],
    package_data={
        # Include configuration files for loggers and experiments
        "dsne_pytorch": ["configs/*.cfg", "configs/*.json"],
    },

    # PyPI metadata
    author="Joshua Newton",
    author_email="jnewt@uvic.ca",
    description="A PyTorch port of the 'd-SNE' domain adaptation approach",
    keywords="d-sne, dsne, domain adaptation, machine learning, deep learning",
    project_urls={
        "Source Code": "https://github.com/joshuacwnewton/d-SNE-PyTorch",
    },
)
