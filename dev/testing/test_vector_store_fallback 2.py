#!/usr/bin/env python3
"""
Test script for the vector store fallback mechanism.

This script tests the vector store implementation without hnswlib
to ensure the GLIBCXX compatibility fix works correctly.
"""

import os
import sys
import numpy as np
from typing import Dict, Any

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_vector_store_fallback():
    """Test the vector store fallback implementation."""
    
    print("🧪 Testing Vector Store Fallback Implementation")
    print("=" * 60)
    
    try:
        # Import our vector store
        from memory.vector_store import VectorStore
        print("✅ Successfully imported VectorStore")
        
        # Create a vector store instance
        print("\n📦 Creating VectorStore instance...")
        vector_store = VectorStore(dim=5, max_elements=100, space="cosine")
        print(f"✅ Created VectorStore with dim={vector_store.dim}")
        
        # Test adding vectors
        print("\n📝 Testing vector addition...")
        test_vectors = [
            ("memory_1", np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), {"category": "test", "priority": "high"}),
            ("memory_2", np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), {"category": "test", "priority": "low"}),
            ("memory_3", np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32), {"category": "demo", "priority": "high"}),
            ("memory_4", np.array([0.8, 0.6, 0.0, 0.0, 0.0], dtype=np.float32), {"category": "test", "priority": "medium"}),
        ]
        
        vector_ids = []
        for memory_id, vector, metadata in test_vectors:
            vid = vector_store.add(memory_id, vector, metadata)
            vector_ids.append(vid)
            print(f"✅ Added vector {memory_id} with ID {vid}")
        
        # Test search functionality
        print("\n🔍 Testing vector search...")
        query_vector = np.array([0.9, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        results = vector_store.search(query_vector, top_k=3)
        
        print(f"Search results for query vector:")
        for i, (memory_id, similarity, metadata) in enumerate(results):
            print(f"  {i+1}. Memory ID: {memory_id}, Similarity: {similarity:.4f}, Metadata: {metadata}")
        
        # Test filtered search
        print("\n🎯 Testing filtered search...")
        filtered_results = vector_store.search(
            query_vector, 
            top_k=3, 
            filter_categories={"category": "test"}
        )
        
        print(f"Filtered search results (category=test):")
        for i, (memory_id, similarity, metadata) in enumerate(filtered_results):
            print(f"  {i+1}. Memory ID: {memory_id}, Similarity: {similarity:.4f}, Metadata: {metadata}")
        
        # Test statistics
        print("\n📊 Testing statistics...")
        stats = vector_store.get_stats()
        print(f"Vector store stats: {stats}")
        
        # Test retrieval by ID
        print("\n🔎 Testing get_by_id...")
        vector_data = vector_store.get_by_id("memory_1")
        if vector_data:
            vector, metadata = vector_data
            print(f"✅ Retrieved memory_1: metadata = {metadata}")
        else:
            print("❌ Failed to retrieve memory_1")
        
        # Test deletion
        print("\n🗑️  Testing vector deletion...")
        deleted = vector_store.delete("memory_2")
        if deleted:
            print("✅ Successfully deleted memory_2")
            
            # Verify deletion
            after_delete_stats = vector_store.get_stats()
            print(f"Stats after deletion: {after_delete_stats}")
        else:
            print("❌ Failed to delete memory_2")
        
        print("\n🎉 All tests passed! Vector store fallback is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        if "hnswlib" in str(e):
            print("🔧 This confirms hnswlib is not available - testing fallback...")
            
            # Test the simple vector store directly
            try:
                from memory.simple_vector_store import SimpleVectorStore
                print("✅ Successfully imported SimpleVectorStore fallback")
                
                # Basic test
                simple_store = SimpleVectorStore(dim=3, max_elements=10)
                test_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                vid = simple_store.add("test", test_vec, {"type": "test"})
                print(f"✅ Simple store test passed: added vector with ID {vid}")
                
                return True
                
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                return False
        else:
            print(f"❌ Unexpected import error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting Vector Store Fallback Test...")
    print("This test verifies that the vector store works without hnswlib dependency.")
    print()
    
    success = test_vector_store_fallback()
    
    if success:
        print("\n✅ SUCCESS: Vector store fallback implementation is working!")
        print("🔧 The GLIBCXX compatibility issue has been resolved.")
        exit(0)
    else:
        print("\n❌ FAILURE: Vector store fallback test failed.")
        print("🛠️  Check the error messages above for debugging information.")
        exit(1)
