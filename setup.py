#!/usr/bin/env python3
"""
NIS Protocol - Neuro-Inspired System Protocol
Setup configuration for PyPI distribution
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read version from VERSION file
version_file = Path(__file__).parent / "system" / "docs" / "VERSION"
version = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else "3.2.1"

setup(
    name="nis-protocol",
    version=version,
    author="Organica AI Solutions",
    author_email="contact@organicaai.com",
    description="Neuro-Inspired System Protocol: Autonomous AI framework with physics validation, multi-agent orchestration, and LLM integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Organica-Ai-Solutions/NIS_Protocol",
    project_urls={
        "Homepage": "https://www.organicaai.com",
        "Documentation": "https://github.com/Organica-Ai-Solutions/NIS_Protocol/tree/main/system/docs",
        "Source": "https://github.com/Organica-Ai-Solutions/NIS_Protocol",
        "Tracker": "https://github.com/Organica-Ai-Solutions/NIS_Protocol/issues",
        "Whitepaper": "https://github.com/Organica-Ai-Solutions/NIS_Protocol/blob/main/system/docs/NIS_Protocol_V3_Whitepaper.md",
    },
    packages=find_packages(where=".", include=["src*", "nis_protocol*"]),
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.6.0",
        "python-multipart>=0.0.9",
        
        # LLM Providers
        "openai>=1.12.0",
        "anthropic>=0.18.1",
        "google-generativeai>=0.4.0",
        
        # AI/ML
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        
        # LangChain & LangGraph
        "langchain>=0.1.9",
        "langchain-core>=0.1.27",
        "langgraph>=0.0.20",
        "langsmith>=0.1.0",
        
        # Voice & Audio
        "gtts>=2.5.0",
        "openai-whisper>=20231117",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
        "pydub>=0.25.1",
        
        # Web & APIs
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "aiohttp>=3.9.0",
        
        # Data & Storage
        "redis>=5.0.0",
        "kafka-python>=2.0.2",
        
        # Utilities
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "click>=8.1.0",
    ],
    extras_require={
        # Development dependencies
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pre-commit>=3.6.0",
        ],
        
        # Full installation with all optional features
        "full": [
            "suno-bark>=0.1.0",  # Bark TTS
            "transformers>=4.37.0",  # Hugging Face models
            "einops>=0.7.0",
            "encodec>=0.1.1",
            "nltk>=3.8.1",
            "boto3>=1.34.0",  # AWS integration
            "pillow>=10.2.0",  # Image processing
            "matplotlib>=3.8.0",  # Visualization
            "pandas>=2.2.0",  # Data analysis
        ],
        
        # Edge deployment (BitNet, low-power)
        "edge": [
            "onnx>=1.15.0",
            "onnxruntime>=1.17.0",
        ],
        
        # Drone/robotics integration
        "drone": [
            "pyserial>=3.5",
            "pymavlink>=2.4.0",
        ],
        
        # Documentation
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.5.0",
            "mkdocstrings[python]>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nis-protocol=nis_protocol.cli:main",
            "nis-server=nis_protocol.server:run",
            "nis-agent=nis_protocol.agent:run",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    keywords=[
        "ai", "agi", "multi-agent", "autonomous", "llm", "machine-learning",
        "physics-informed", "kan", "pinn", "neuro-inspired", "langchain",
        "langgraph", "cognitive-architecture", "consciousness", "reasoning",
        "drone", "robotics", "edge-ai", "autonomous-systems"
    ],
    include_package_data=True,
    zip_safe=False,
)
