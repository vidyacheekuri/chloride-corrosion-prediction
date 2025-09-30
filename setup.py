"""
Setup script for Chloride Corrosion Prediction project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chloride-corrosion-prediction",
    version="1.0.0",
    author="Research Assistant",
    author_email="research@example.com",
    description="A comprehensive machine learning pipeline for predicting chloride-induced corrosion rates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chloride-corrosion-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "chloride-predict=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.pkl", "*.png"],
    },
)
