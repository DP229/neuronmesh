from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuronmesh",
    version="0.1.0",
    author="NeuronMesh Team",
    author_email="hello@neuronmesh.dev",
    description="Distributed intelligent autoagent platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dp229/neuronmesh",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "black>=23.0",
            "ruff>=0.1.0",
        ],
        "openai": [
            "openai>=1.0",
        ],
        "anthropic": [
            "anthropic>=0.8",
        ],
        "redis": [
            "redis>=5.0",
        ],
        "vector": [
            "qdrant-client>=1.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuronmesh=neuronmesh_cli.main:main",
        ],
    },
)
