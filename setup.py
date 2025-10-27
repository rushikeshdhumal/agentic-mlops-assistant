"""Setup script for ATHENA MLOps Platform."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="athena-mlops",
    version="0.1.0",
    author="Rushikesh Dhumal",
    author_email="your.email@example.com",
    description="Agentic MLOps assistant with natural language interface for edge AI optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rushikeshdhumal/agentic-mlops-assistant",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in (this_directory / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ],
    extras_require={
        "dev": [
            line.strip()
            for line in (this_directory / "requirements-dev.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#") and not line.startswith("-r")
        ],
    },
    entry_points={
        "console_scripts": [
            "athena=athena.__main__:main",
        ],
    },
)
