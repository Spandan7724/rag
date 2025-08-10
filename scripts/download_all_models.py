#!/usr/bin/env python3
"""
Download and cache ALL models used in the RAG application
This includes embedding models, reranker models, and any other AI models
"""
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

# Configuration for all models used in the application
MODELS_CONFIG = {
    "embedding": {
        "name": "BGE-M3 Embedding Model",
        "model_id": "BAAI/bge-m3",
        "type": "sentence_transformers",
        "description": "Multi-functional embedding model for dense retrieval, sparse retrieval, and multi-vector interaction"
    },
    "reranker": {
        "name": "BGE Reranker Large",
        "model_id": "BAAI/bge-reranker-large", 
        "type": "cross_encoder",
        "description": "Cross-encoder reranking model for improved accuracy"
    }
}

def log_message(message: str):
    """Log with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def download_sentence_transformer(model_id: str, model_name: str) -> bool:
    """Download a SentenceTransformer model"""
    try:
        log_message(f"Downloading {model_name} ({model_id})...")
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id)
        
        # Test the model
        log_message(f"Testing {model_name}...")
        test_embedding = model.encode(["Test sentence for verification"])
        
        log_message(f"{model_name} downloaded and verified! Shape: {test_embedding.shape}")
        return True
        
    except Exception as e:
        log_message(f"Failed to download {model_name}: {e}")
        return False

def download_cross_encoder(model_id: str, model_name: str) -> bool:
    """Download a CrossEncoder model"""
    try:
        log_message(f"Downloading {model_name} ({model_id})...")
        
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_id)
        
        # Test the model
        log_message(f"Testing {model_name}...")
        test_score = model.predict([("test query", "test document")])
        
        log_message(f"{model_name} downloaded and verified! Test score: {test_score}")
        return True
        
    except Exception as e:
        log_message(f"Failed to download {model_name}: {e}")
        return False

def main():
    """Download all models used in the application"""
    log_message("Starting download of ALL models for RAG application...")
    
    # Ensure cache directory exists
    cache_dir = Path(os.environ.get('HF_HOME', '/home/appuser/.cache/huggingface'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_message(f"Using cache directory: {cache_dir}")
    
    # Track download results
    results = {}
    total_models = len(MODELS_CONFIG)
    
    log_message(f"Will download {total_models} models:")
    for key, config in MODELS_CONFIG.items():
        log_message(f"  * {config['name']} - {config['description']}")
    
    print("\n" + "="*60)
    
    # Download each model
    for i, (key, config) in enumerate(MODELS_CONFIG.items(), 1):
        model_name = config['name']
        model_id = config['model_id']
        model_type = config['type']
        
        log_message(f"[{i}/{total_models}] Processing {model_name}...")
        
        start_time = time.time()
        
        if model_type == "sentence_transformers":
            success = download_sentence_transformer(model_id, model_name)
        elif model_type == "cross_encoder":
            success = download_cross_encoder(model_id, model_name)
        else:
            log_message(f"Unknown model type: {model_type}")
            success = False
        
        download_time = time.time() - start_time
        results[key] = {
            'success': success,
            'time': download_time,
            'name': model_name
        }
        
        if success:
            log_message(f"Downloaded {model_name} in {download_time:.1f}s")
        
        print()  # Add spacing between models
    
    # Summary
    print("="*60)
    log_message("Download Summary:")
    
    successful = [k for k, v in results.items() if v['success']]
    failed = [k for k, v in results.items() if not v['success']]
    
    for key in successful:
        result = results[key]
        log_message(f"SUCCESS: {result['name']} - {result['time']:.1f}s")
    
    for key in failed:
        result = results[key]
        log_message(f"FAILED: {result['name']}")
    
    total_time = sum(r['time'] for r in results.values())
    log_message(f"Total download time: {total_time:.1f}s")
    
    if len(successful) == total_models:
        log_message("ALL MODELS DOWNLOADED SUCCESSFULLY!")
        log_message("The application is ready for fast startup!")
        return True
    else:
        log_message(f"Downloaded {len(successful)}/{total_models} models")
        log_message("Some models failed to download")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
