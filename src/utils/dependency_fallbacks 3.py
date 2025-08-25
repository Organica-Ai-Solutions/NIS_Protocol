#!/usr/bin/env python3
"""
Robust dependency fallback system for NIS Protocol v3.2
Provides graceful fallbacks for ML and advanced dependencies
"""

import logging
import hashlib
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DependencyStatus:
    """Track status of optional dependencies"""
    name: str
    available: bool
    version: Optional[str] = None
    error: Optional[str] = None
    fallback_active: bool = False

class DependencyManager:
    """Centralized dependency management with fallbacks"""
    
    def __init__(self):
        self.dependencies = {}
        self.fallback_providers = {}
        self._check_all_dependencies()
    
    def _check_all_dependencies(self):
        """Check availability of all optional dependencies"""
        
        # Check sentence-transformers
        self._check_dependency("sentence_transformers", "sentence-transformers")
        
        # Check NVIDIA NeMo
        self._check_dependency("nemo", "nemo-toolkit")
        
        # Check additional ML packages
        self._check_dependency("optimum", "optimum")
        self._check_dependency("peft", "peft")
        self._check_dependency("hnswlib", "hnswlib")
        
        # Check web search dependencies
        self._check_dependency("arxiv", "arxiv")
        self._check_dependency("selenium", "selenium")
        
        # Log summary
        available = sum(1 for dep in self.dependencies.values() if dep.available)
        total = len(self.dependencies)
        logger.info(f"Dependency check: {available}/{total} optional packages available")
    
    def _check_dependency(self, module_name: str, package_name: str = None):
        """Check if a specific dependency is available"""
        package_name = package_name or module_name
        
        try:
            __import__(module_name)
            # Try to get version
            try:
                import importlib.metadata
                version = importlib.metadata.version(package_name)
            except:
                version = "unknown"
            
            self.dependencies[module_name] = DependencyStatus(
                name=package_name,
                available=True,
                version=version
            )
            logger.debug(f"✅ {package_name} available (v{version})")
            
        except ImportError as e:
            self.dependencies[module_name] = DependencyStatus(
                name=package_name,
                available=False,
                error=str(e),
                fallback_active=True
            )
            logger.debug(f"⚠️ {package_name} not available: {e}")
    
    def is_available(self, module_name: str) -> bool:
        """Check if a dependency is available"""
        return self.dependencies.get(module_name, DependencyStatus("", False)).available
    
    def get_status(self, module_name: str) -> DependencyStatus:
        """Get detailed status of a dependency"""
        return self.dependencies.get(module_name, DependencyStatus(module_name, False))
    
    def register_fallback(self, module_name: str, fallback_provider):
        """Register a fallback provider for a missing dependency"""
        self.fallback_providers[module_name] = fallback_provider
        logger.debug(f"Registered fallback for {module_name}")
    
    def get_provider(self, module_name: str):
        """Get the real module or fallback provider"""
        if self.is_available(module_name):
            return __import__(module_name)
        else:
            return self.fallback_providers.get(module_name)

# Global dependency manager instance
dependency_manager = DependencyManager()

# ===== SENTENCE TRANSFORMERS FALLBACK =====

class FallbackSentenceTransformer:
    """Fallback implementation for SentenceTransformer"""
    
    def __init__(self, model_name: str = "fallback", *args, **kwargs):
        self.model_name = model_name
        self.embedding_dim = 384  # Default embedding dimension
        logger.warning(f"Using fallback SentenceTransformer for {model_name}")
    
    def encode(self, texts: Union[str, List[str]], *args, **kwargs) -> List[List[float]]:
        """Generate hash-based embeddings as fallback"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Create deterministic hash-based embedding
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            
            # Convert to normalized float vector
            embedding = []
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i:i+4]
                if len(chunk) == 4:
                    # Convert 4 bytes to float and normalize
                    val = int.from_bytes(chunk, 'big')
                    normalized = (val / (2**32)) * 2 - 1  # Normalize to [-1, 1]
                    embedding.append(normalized)
            
            # Pad or truncate to desired dimension
            while len(embedding) < self.embedding_dim:
                embedding.append(0.0)
            embedding = embedding[:self.embedding_dim]
            
            embeddings.append(embedding)
        
        return embeddings
    
    @property
    def device(self):
        return "cpu"
    
    def to(self, device):
        return self

# Register fallback
if not dependency_manager.is_available("sentence_transformers"):
    dependency_manager.register_fallback("sentence_transformers", type('Module', (), {
        'SentenceTransformer': FallbackSentenceTransformer
    }))

# ===== NVIDIA NEMO FALLBACK =====

class FallbackNeMoModel:
    """Fallback implementation for NeMo models"""
    
    def __init__(self, model_name: str = "fallback_nemo", *args, **kwargs):
        self.model_name = model_name
        logger.warning(f"Using fallback NeMo model for {model_name}")
    
    def generate(self, inputs: Union[str, List[str]], *args, **kwargs) -> List[str]:
        """Generate fallback responses"""
        if isinstance(inputs, str):
            inputs = [inputs]
        
        responses = []
        for input_text in inputs:
            response = f"Fallback NeMo response for: {input_text[:50]}..."
            responses.append(response)
        
        return responses
    
    def forward(self, *args, **kwargs):
        return {"logits": [[0.1, 0.2, 0.7]], "predictions": ["fallback_prediction"]}

class FallbackNeMoAgentToolkit:
    """Fallback implementation for NeMo Agent Toolkit"""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using fallback NeMo Agent Toolkit")
    
    def create_agent(self, agent_type: str, **kwargs):
        return FallbackNeMoAgent(agent_type)
    
    def orchestrate(self, agents: List, workflow: str, **kwargs):
        return {
            "status": "fallback_orchestration",
            "agents": len(agents),
            "workflow": workflow,
            "result": "Minimal coordination completed"
        }

class FallbackNeMoAgent:
    """Fallback implementation for individual NeMo agents"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.status = "fallback_active"
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "status": "processed_fallback",
            "result": f"Fallback processing for {input_data}",
            "timestamp": time.time()
        }

# Register NeMo fallbacks
if not dependency_manager.is_available("nemo"):
    nemo_fallback = type('Module', (), {
        'Model': FallbackNeMoModel,
        'AgentToolkit': FallbackNeMoAgentToolkit,
        'Agent': FallbackNeMoAgent
    })
    dependency_manager.register_fallback("nemo", nemo_fallback)

# ===== HUGGING FACE OPTIMIZATIONS FALLBACK =====

class FallbackOptimumModel:
    """Fallback for Optimum accelerated models"""
    
    def __init__(self, model_name: str, *args, **kwargs):
        self.model_name = model_name
        logger.warning(f"Using fallback Optimum model for {model_name}")
    
    def generate(self, *args, **kwargs):
        return {"generated_text": "Fallback optimized generation"}
    
    def __call__(self, *args, **kwargs):
        return {"output": "Fallback model output"}

if not dependency_manager.is_available("optimum"):
    dependency_manager.register_fallback("optimum", type('Module', (), {
        'Model': FallbackOptimumModel
    }))

# ===== HNSWLIB FALLBACK =====

class FallbackHNSWIndex:
    """Fallback vector index using simple dictionary storage"""
    
    def __init__(self, space: str = 'cosine', dim: int = 384):
        self.space = space
        self.dim = dim
        self.data = {}
        self.labels = []
        self.current_count = 0
        logger.warning("Using fallback HNSW index with simple storage")
    
    def init_index(self, max_elements: int):
        self.max_elements = max_elements
    
    def add_items(self, vectors: List[List[float]], labels: List[int] = None):
        if labels is None:
            labels = list(range(self.current_count, self.current_count + len(vectors)))
        
        for vector, label in zip(vectors, labels):
            self.data[label] = vector
            if label not in self.labels:
                self.labels.append(label)
        
        self.current_count += len(vectors)
    
    def knn_query(self, query_vector: List[float], k: int = 5):
        """Simple cosine similarity search"""
        if not self.data:
            return [], []
        
        similarities = []
        for label, vector in self.data.items():
            # Simple cosine similarity
            dot_product = sum(a * b for a, b in zip(query_vector, vector))
            norm_a = sum(a * a for a in query_vector) ** 0.5
            norm_b = sum(b * b for b in vector) ** 0.5
            
            if norm_a > 0 and norm_b > 0:
                similarity = dot_product / (norm_a * norm_b)
                similarities.append((similarity, label))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        similarities = similarities[:k]
        
        labels_result = [label for _, label in similarities]
        distances = [1 - sim for sim, _ in similarities]  # Convert to distances
        
        return labels_result, distances

if not dependency_manager.is_available("hnswlib"):
    dependency_manager.register_fallback("hnswlib", type('Module', (), {
        'Index': FallbackHNSWIndex
    }))

# ===== ARXIV FALLBACK =====

class FallbackArxivClient:
    """Fallback ArXiv client with sample data"""
    
    def __init__(self):
        logger.warning("Using fallback ArXiv client")
    
    def search(self, query: str, max_results: int = 10):
        """Return sample paper data"""
        papers = []
        for i in range(min(max_results, 3)):
            papers.append({
                "title": f"Sample Paper {i+1}: {query}",
                "authors": ["Sample Author", "Another Author"],
                "abstract": f"This is a sample abstract for research on {query}. "
                           "This would normally be a real paper from ArXiv.",
                "url": f"https://arxiv.org/abs/sample.{i+1}",
                "published": time.strftime("%Y-%m-%d"),
                "categories": ["cs.AI", "cs.LG"]
            })
        return papers

if not dependency_manager.is_available("arxiv"):
    dependency_manager.register_fallback("arxiv", type('Module', (), {
        'Client': FallbackArxivClient
    }))

# ===== UTILITY FUNCTIONS =====

def get_safe_import(module_name: str, fallback_name: str = None):
    """Safely import a module with fallback"""
    fallback_name = fallback_name or module_name
    
    if dependency_manager.is_available(module_name):
        return __import__(module_name)
    else:
        fallback = dependency_manager.get_provider(module_name)
        if fallback:
            logger.debug(f"Using fallback for {module_name}")
            return fallback
        else:
            raise ImportError(f"No fallback available for {module_name}")

def check_ml_readiness() -> Dict[str, Any]:
    """Check overall ML ecosystem readiness"""
    ml_deps = ["sentence_transformers", "nemo", "optimum", "hnswlib"]
    
    status = {
        "ready_for_production": True,
        "available_dependencies": [],
        "missing_dependencies": [],
        "fallback_active": [],
        "recommendations": []
    }
    
    for dep in ml_deps:
        dep_status = dependency_manager.get_status(dep)
        if dep_status.available:
            status["available_dependencies"].append(dep)
        else:
            status["missing_dependencies"].append(dep)
            status["fallback_active"].append(dep)
            status["ready_for_production"] = False
    
    # Generate recommendations
    if status["missing_dependencies"]:
        status["recommendations"].append(
            f"Install missing packages: {', '.join(status['missing_dependencies'])}"
        )
    
    if not status["ready_for_production"]:
        status["recommendations"].append(
            "System running in fallback mode - install ML dependencies for full features"
        )
    
    return status

def get_dependency_report() -> Dict[str, Any]:
    """Generate comprehensive dependency report"""
    return {
        "timestamp": time.time(),
        "dependencies": {
            name: {
                "available": status.available,
                "version": status.version,
                "fallback_active": status.fallback_active,
                "error": status.error
            }
            for name, status in dependency_manager.dependencies.items()
        },
        "ml_readiness": check_ml_readiness(),
        "fallback_providers": list(dependency_manager.fallback_providers.keys())
    }

# ===== INITIALIZATION =====

def initialize_fallbacks():
    """Initialize all fallback systems"""
    logger.info("🔧 Initializing dependency fallback system")
    
    # Check and log status
    report = get_dependency_report()
    ml_status = report["ml_readiness"]
    
    if ml_status["ready_for_production"]:
        logger.info("✅ All ML dependencies available - full functionality enabled")
    else:
        missing = len(ml_status["missing_dependencies"])
        total = len(ml_status["missing_dependencies"]) + len(ml_status["available_dependencies"])
        logger.warning(f"⚠️ {missing}/{total} ML dependencies missing - fallback mode active")
        
        for rec in ml_status["recommendations"]:
            logger.info(f"💡 {rec}")
    
    return dependency_manager

# Initialize on import
initialize_fallbacks()

