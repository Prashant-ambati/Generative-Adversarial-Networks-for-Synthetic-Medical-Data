"""
Setup script for Synthetic Medical Data GAN
Built by Prashant Ambati
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="synthetic-medical-gan",
    version="1.0.0",
    author="Prashant Ambati",
    author_email="prashant.ambati@example.com",
    description="Generative Adversarial Networks for Privacy-Preserving Synthetic Medical Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prashantambati/synthetic-medical-gan",
    project_urls={
        "Bug Tracker": "https://github.com/prashantambati/synthetic-medical-gan/issues",
        "Documentation": "https://github.com/prashantambati/synthetic-medical-gan#readme",
        "Source Code": "https://github.com/prashantambati/synthetic-medical-gan",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "notebook": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-medical-gan=train_gan:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)