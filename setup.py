from setuptools import setup, find_packages

setup(
    name="mlreserving",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "nnetsauce",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
        ],
    },
    python_requires=">=3.7",
    author="T. Moudiki",
    author_email="thierry.moudiki@gmail.com",
    description="Model-agnostic Probabilistic Machine Learning Reserving",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Techtonique/mlreserving",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)