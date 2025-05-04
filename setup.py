from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nis-protocol",
    version="0.1.0",
    author="Diego Torres",
    author_email="contact@organicaai.com",
    description="Neuro-Inspired System Protocol for intelligent multi-agent systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OrganicaAI/NIS-Protocol",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "redis>=6.0.0",
        "hiredis>=2.0.0",
        "pydantic>=1.9.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "httpx>=0.23.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
    ],
    extras_require={
        "vision": [
            "opencv-python>=4.6.0",
            "pillow>=9.4.0",
        ],
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
        ],
    },
) 