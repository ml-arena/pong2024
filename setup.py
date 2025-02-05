from setuptools import setup, find_packages

setup(
    name="pong2024",
    version="0.1.0",
    description="ML-Arena: 2-Player Pong 2024 Competition Environment",
    author="ML-Arena Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "gymnasium>=0.29.1",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    python_requires=">=3.10",
)