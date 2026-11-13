"""Package configuration for Explainable Medical Image Classification System."""

from setuptools import setup, find_packages

setup(
    name="chestxray_xai",
    version="1.0.0",
    author="Expert AI Engineering Team",
    description="An Explainable Deep Learning Framework for Multi-Label Chest X-Ray Disease Classification Using CNNs and Grad-CAM",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "streamlit>=1.28.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
