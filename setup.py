"""Setup configuration for bee-monitor package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="bee-monitor",
    version="1.0.0",
    author="Edward Amoah",
    author_email="your.email@example.com",
    description="Computer vision system for monitoring solitary bee activity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bee-monitor",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/bee-monitor/issues",
        "Documentation": "https://github.com/yourusername/bee-monitor/docs",
        "Source Code": "https://github.com/yourusername/bee-monitor",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "pylint>=2.17",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.2",
            "myst-parser>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bee-monitor=bee_monitor.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bee_monitor": ["config/*.yaml"],
    },
    zip_safe=False,
)
