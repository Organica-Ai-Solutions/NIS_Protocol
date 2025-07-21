from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nis-protocol",
    version="3.0.0",
    author="Diego Torres",
    author_email="contact@organicaai.com",
    description="NIS Protocol v3.0 - Advanced Multi-Agent System with LangGraph/LangSmith Integration and Physics-Informed Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Organica-Ai-Solutions/NIS_Protocol",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
        "Topic :: Scientific/Engineering :: Physics",
        "Framework :: AsyncIO",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core infrastructure
        "redis>=6.0.0",
        "hiredis>=2.0.0",
        "pydantic>=1.9.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "httpx>=0.23.0",
        
        # Scientific computing
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        
        # Environment and configuration
        "python-dotenv>=1.0.0",
        
        # Async and concurrency
        "asyncio-mqtt>=0.16.0",
        "aiofiles>=23.0.0",
        
        # Data handling
        "pandas>=1.5.0",
        "jsonschema>=4.0.0",
    ],
    extras_require={
        "full": [
            # LangChain ecosystem
            "langchain>=0.1.0",
            "langchain-core>=0.1.0",
            "langchain-community>=0.0.10",
            "langgraph>=0.0.20",
            "langsmith>=0.0.80",
            
            # LLM providers
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "aiohttp>=3.8.0",
            "tiktoken>=0.5.0",
            
            # Infrastructure
            "kafka-python>=2.0.2",
            "aiokafka>=0.8.0",
            
            # Additional ML/AI
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
        "langchain": [
            "langchain>=0.1.0",
            "langchain-core>=0.1.0", 
            "langchain-community>=0.0.10",
            "langgraph>=0.0.20",
            "langsmith>=0.0.80",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "aiohttp>=3.8.0",
            "tiktoken>=0.5.0",
        ],
        "infrastructure": [
            "kafka-python>=2.0.2",
            "aiokafka>=0.8.0",
            "redis[hiredis]>=6.0.0",
        ],
        "vision": [
            "opencv-python>=4.6.0",
            "pillow>=9.4.0",
        ],
        "dev": [
            "pytest>=7.3.1",
            "pytest-asyncio>=0.21.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nis-setup=scripts.setup_env_example:main",
            "nis-test=scripts.test_env_config:main",
        ],
    },
    keywords="ai, agents, consciousness, multi-agent, llm, cognitive-architecture, agi, langgraph, langsmith, physics-informed",
    project_urls={
        "Documentation": "https://organica-ai-solutions.github.io/NIS_Protocol/",
        "Source": "https://github.com/Organica-Ai-Solutions/NIS_Protocol",
        "Bug Reports": "https://github.com/Organica-Ai-Solutions/NIS_Protocol/issues",
        "Funding": "https://github.com/sponsors/Organica-Ai-Solutions",
    },
) 