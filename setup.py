"""
NIS Protocol v3.2.1 - Production-Ready AI Development Platform
==============================================================

🚀 **The Complete AI Operating System for Modern Applications**

NIS Protocol is a production-ready AI development platform that enables developers to build 
sophisticated AI applications with physics validation, multi-agent coordination, and 
enterprise-grade deployment capabilities.

## 🎯 **Key Features**

### **🤖 Multi-Agent AI System**
- **LLM Integration**: OpenAI, Anthropic, Google, DeepSeek, NVIDIA, BitNet support
- **Agent Coordination**: Intelligent task distribution and collaboration
- **Real-time Processing**: Sub-second response times with streaming support

### **⚡ Physics-Informed AI**
- **PINN Integration**: Physics-Informed Neural Networks for scientific computing
- **KAN Networks**: Kolmogorov-Arnold Networks for symbolic reasoning
- **Validation Engine**: Automatic physics constraint checking and correction

### **🌐 Production Deployment**
- **Docker Ready**: Complete containerization with docker-compose
- **API Gateway**: FastAPI-based REST API with automatic documentation
- **Web Interface**: Modern chat interfaces with LangChain agent visualization
- **Security**: Enterprise-grade authentication, encryption, and audit logging

### **🔧 Developer Experience**
- **Easy Installation**: `pip install nis-protocol-v321-organica-enhanced`
- **Rich Documentation**: Comprehensive guides and API references
- **Live Examples**: Ready-to-run demos and tutorials
- **GitHub Pages**: https://nisprotocol.organicaai.com/

## 📦 **Installation Options**

```bash
# Full platform installation
pip install nis-protocol-v321-organica-enhanced[platform]

# Edge devices (Raspberry Pi, embedded)
pip install nis-protocol-v321-organica-enhanced[edge]

# Cloud deployment
pip install nis-protocol-v321-organica-enhanced[cloud]

# Minimal installation
pip install nis-protocol-v321-organica-enhanced[minimal]
```

## 🚀 **Quick Start**

```python
from nis_protocol import NISPlatform

# Initialize the platform
platform = NISPlatform()

# Start the AI agents
await platform.start_agents()

# Process with physics validation
result = await platform.process_with_physics(
    query="Solve heat equation",
    physics_constraints={"temperature_bounds": [0, 100]}
)

print(f"Result: {result.solution}")
print(f"Physics Compliance: {result.physics_score}")
```

## 🌟 **Use Cases**

- **🏭 Industrial IoT**: Smart manufacturing and predictive maintenance
- **🚁 Autonomous Systems**: Drones, robotics, and self-driving vehicles  
- **🏠 Smart Infrastructure**: Building automation and energy management
- **🔬 Scientific Computing**: Physics simulations and research applications
- **☁️ Enterprise AI**: Scalable AI services and microservices architecture

## 📚 **Resources**

- **Documentation**: https://nisprotocol.organicaai.com/
- **GitHub**: https://github.com/Organica-Ai-Solutions/NIS_Protocol
- **API Reference**: Auto-generated FastAPI docs at `/docs`
- **Examples**: Complete working examples in `/examples` directory

Built with ❤️ by Organica AI Solutions
"""

from setuptools import setup, find_packages
import os

# Read version from VERSION file
def get_version():
    try:
        with open("VERSION", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "3.2.1"  # Updated fallback to match VERSION file

# Read long description from README
def get_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return __doc__

# Read requirements from requirements.txt
def get_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "fastapi>=0.110.0",
            "uvicorn[standard]>=0.20.0",
            "pydantic>=2.0.0",
            "numpy>=1.24.0",
            "torch>=2.0.0",
            "redis>=4.5.0",
            "aiohttp>=3.8.0",
        ]

setup(
    # Package Identity
    name="nis-protocol-v321-organica-enhanced",
    version=get_version(),
    author="Organica AI Solutions",
    author_email="developers@organicaai.com",
    description="🚀 Production-Ready AI Development Platform with Physics Validation, Multi-Agent Coordination & Enterprise Deployment",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # URLs and Links
    url="https://github.com/Organica-Ai-Solutions/NIS_Protocol",
    project_urls={
        "Documentation": "https://nisprotocol.organicaai.com/",
        "Homepage": "https://nisprotocol.organicaai.com/",
        "Source Code": "https://github.com/Organica-Ai-Solutions/NIS_Protocol",
        "Bug Reports": "https://github.com/Organica-Ai-Solutions/NIS_Protocol/issues",
        "Pull Requests": "https://github.com/Organica-Ai-Solutions/NIS_Protocol/pulls",
        "Releases": "https://github.com/Organica-Ai-Solutions/NIS_Protocol/releases",
        "Examples": "https://github.com/Organica-Ai-Solutions/NIS_Protocol/tree/main/examples",
        "Docker Hub": "https://hub.docker.com/r/organicaai/nis-protocol",
        "PyPI Package": "https://test.pypi.org/project/nis-protocol-v321-organica-enhanced/",
        "Live Demo": "https://nisprotocol.organicaai.com/",
    },
    
    # Package Structure
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    
    # Platform Classifications
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",
        
        # Intended Audience  
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Information Technology",
        "Intended Audience :: System Administrators",
        
        # License
        "License :: OSI Approved :: Apache Software License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Topic Categories - Platform Focus
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: System :: Operating System",
        "Topic :: System :: Distributed Computing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Home Automation",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Multimedia :: Graphics :: Presentation",
        "Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator",
        
        # Environment
        "Environment :: Console",
        "Environment :: Web Environment",
        "Environment :: No Input/Output (Daemon)",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        
        # Frameworks
        "Framework :: AsyncIO",
        "Framework :: FastAPI",
    ],
    
    # Keywords for PyPI search discoverability
    keywords=[
        # Core Technology
        "ai", "artificial-intelligence", "machine-learning", "deep-learning",
        "neural-networks", "llm", "large-language-models", "multi-agent",
        
        # Physics & Science
        "physics-informed", "pinn", "kan", "kolmogorov-arnold", "differential-equations",
        "scientific-computing", "numerical-methods", "physics-validation",
        
        # Platform & Framework
        "platform", "framework", "sdk", "api", "fastapi", "asyncio", "microservices",
        "containerization", "docker", "production-ready", "enterprise",
        
        # Applications
        "robotics", "autonomous-systems", "iot", "edge-computing", "drones", "uav",
        "smart-infrastructure", "industrial-automation", "predictive-maintenance",
        
        # Integration
        "openai", "anthropic", "google-ai", "langchain", "langgraph", "streaming",
        "real-time", "multi-provider", "hybrid-cloud", "edge-deployment",
        
        # Development
        "developer-tools", "rapid-prototyping", "production-deployment",
        "monitoring", "logging", "security", "authentication", "encryption"
    ],
    
    # Python Requirements
    python_requires=">=3.8",
    
    # Core Dependencies
    install_requires=get_requirements(),
    
    # Optional Feature Sets
    extras_require={
        # ==== PLATFORM EDITIONS ====
        "platform": [
            # Full platform with all capabilities
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.35.0",
            "langchain>=0.1.0",
            "langgraph>=0.1.0",
            "sentence-transformers>=2.2.0",
            "opencv-python>=4.8.0",
            "pillow>=10.0.0",
        ],
        
        "edge": [
            # Optimized for edge devices (Raspberry Pi, embedded systems)
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "opencv-python-headless>=4.8.0",
            "psutil>=5.9.0",
        ],
        
        "cloud": [
            # Full cloud capabilities with all providers
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "langchain>=0.1.0",
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "google-cloud-aiplatform>=1.38.0",
        ],
        
        "minimal": [
            # Minimal installation for basic functionality
            "torch>=2.0.0",
            "numpy>=1.24.0",
        ],
        
        # ==== USE CASE SPECIFIC ====
        "robotics": [
            # Robotics and autonomous systems
            "torch>=2.0.0",
            "opencv-python>=4.8.0",
            "scipy>=1.10.0",
            "matplotlib>=3.7.0",
            "pyserial>=3.5",
        ],
        
        "drone": [
            # Drone and UAV systems
            "torch>=2.0.0", 
            "opencv-python>=4.8.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "pyserial>=3.5",
            "gps>=3.20",
        ],
        
        "iot": [
            # IoT and sensor networks
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "paho-mqtt>=1.6.0",
            "influxdb-client>=1.36.0",
        ],
        
        "city": [
            # Smart city infrastructure
            "torch>=2.0.0",
            "opencv-python>=4.8.0",
            "postgresql>=0.1.0",
            "geopandas>=0.13.0",
            "folium>=0.14.0",
        ],
        
        "industrial": [
            # Industrial automation
            "torch>=2.0.0",
            "opencv-python>=4.8.0",
            "modbus-tk>=1.1.2",
            "opcua>=0.98.13",
        ],
        
        # ==== CAPABILITY MODULES ====
        "physics": [
            # Physics-informed neural networks
            "torch>=2.0.0",
            "scipy>=1.10.0",
            "sympy>=1.12",
            "matplotlib>=3.7.0",
        ],
        
        "vision": [
            # Computer vision capabilities
            "opencv-python>=4.8.0",
            "pillow>=10.0.0",
            "torchvision>=0.15.0",
            "albumentations>=1.3.0",
        ],
        
        "nlp": [
            # Natural language processing
            "transformers>=4.35.0",
            "sentence-transformers>=2.2.0",
            "spacy>=3.6.0",
            "nltk>=3.8.0",
        ],
        
        "multimodal": [
            # Multimodal AI capabilities
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "opencv-python>=4.8.0",
            "pillow>=10.0.0",
        ],
        
        # ==== INTEGRATION PROTOCOLS ====
        "protocols": [
            # Third-party protocol integration
            "langchain>=0.1.0",
            "langgraph>=0.1.0",
            "grpcio>=1.56.0",
            "protobuf>=4.23.0",
        ],
        
        "nvidia": [
            # NVIDIA ecosystem integration
            "nvidia-ml-py3>=7.352.0",
        ],
        
        # ==== DEVELOPMENT & DEPLOYMENT ====
        "dev": [
            # Development tools
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        
        "deployment": [
            # Deployment and orchestration
            "docker>=6.1.0",
            "kubernetes>=27.2.0",
            "gunicorn>=21.0.0",
            "nginx>=1.1.0",
        ],
        
        "monitoring": [
            # Monitoring and observability
            "prometheus-client>=0.17.0",
            "grafana-api>=1.0.3",
            "structlog>=23.0.0",
        ],
        
        # ==== CONVENIENCE BUNDLES ====
        "all": [
            # Everything - full installation
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "transformers>=4.35.0",
            "langchain>=0.1.0",
            "langgraph>=0.1.0",
            "opencv-python>=4.8.0",
            "sentence-transformers>=2.2.0",
            "scipy>=1.10.0",
            "matplotlib>=3.7.0",
            "nvidia-ml-py3>=7.352.0",
        ],
    },
    
    # Command Line Tools
    entry_points={
        "console_scripts": [
            # Platform Management
            "nis=nis_protocol.cli.main:main",
            "nis-platform=nis_protocol.cli.platform:main",
            "nis-deploy=nis_protocol.cli.deploy:main",
            
            # Development Tools
            "nis-init=nis_protocol.cli.init:main",
            "nis-agent=nis_protocol.cli.agent:main",
            "nis-test=nis_protocol.cli.test:main",
            "nis-serve=nis_protocol.cli.serve:main",
            
            # Device Tools
            "nis-edge=nis_protocol.cli.edge:main",
            "nis-drone=nis_protocol.cli.drone:main",
            "nis-robot=nis_protocol.cli.robot:main",
            
            # Utilities
            "nis-health=nis_protocol.cli.health:main",
            "nis-monitor=nis_protocol.cli.monitor:main",
            "nis-config=nis_protocol.cli.config:main",
        ],
        
        # Plugin System
        "nis_protocol.agents": [
            "consciousness=nis_protocol.agents.consciousness:EnhancedConsciousAgent",
            "physics=nis_protocol.agents.physics:UnifiedPhysicsAgent", 
            "reasoning=nis_protocol.agents.reasoning:UnifiedReasoningAgent",
            "memory=nis_protocol.agents.memory:EnhancedMemoryAgent",
            "vision=nis_protocol.agents.vision:VisionAgent",
        ],
        
        "nis_protocol.protocols": [
            "mcp=nis_protocol.protocols.mcp:MCPAdapter",
            "acp=nis_protocol.protocols.acp:ACPAdapter",
            "a2a=nis_protocol.protocols.a2a:A2AAdapter",
        ],
        
        "nis_protocol.deployment": [
            "docker=nis_protocol.deployment.docker:DockerDeployment",
            "kubernetes=nis_protocol.deployment.k8s:KubernetesDeployment",
            "edge=nis_protocol.deployment.edge:EdgeDeployment",
        ],
    },
    
    # Package Data
    package_data={
        "nis_protocol": [
            "configs/*.json",
            "configs/*.yaml", 
            "templates/*.py",
            "templates/*.json",
            "examples/*.py",
            "docs/*.md",
        ],
    },
    
    # Metadata
    zip_safe=False,
    platforms=["any"],
    license="Apache-2.0",
    
    # Data Files - Only include existing files
    data_files=[
        ("configs", ["configs/protocol_routing.json", "configs/provider_registry.yaml"]),
        ("examples", ["examples/simple_agent.py", "examples/edge_deployment.py"]),
        ("docs", ["README.md", "CHANGELOG.md", "LICENSE"]),
    ],
)
