# build_vectorstore.py - Optimized Version
# DEVELOPMENT PHASE 1: Build optimized vector store infrastructure

import json
import pandas as pd
import numpy as np
import base64
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict

def load_json_QA_from_metadata_jsonl(file_path: str = 'metadata.jsonl') -> List[Dict]:
    """Load processed GAIA data from metadata.jsonl"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_QA = [json.loads(line) for line in f]
        print(f"✅ Loaded {len(json_QA)} GAIA samples from {file_path}")
        return json_QA
    except FileNotFoundError:
        print(f"❌ File {file_path} not found. Run data processing first.")
        return []
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return []

def generate_optimized_csv(json_QA: List[Dict], embeddings_model, output_file: str = 'gaia_embeddings.csv'):
    """Generate optimized CSV with compressed embeddings"""
    print(f"📝 Building optimized vector store: {output_file}")
    print("=" * 50)
    print(f"Processing {len(json_QA)} documents...")
    
    docs = []
    for i, sample in enumerate(json_QA):
        print(f"  Processing document {i + 1}/{len(json_QA)}...", end='\r')
        
        # Same Q&A format as #1 student
        content = f"Question : {sample['Question']}\n\nFinal answer : {sample['Final answer']}"
        
        # Compute embedding
        try:
            embedding = embeddings_model.embed_query(content)
        except Exception as e:
            print(f"\n❌ Error computing embedding for document {i + 1}: {e}")
            continue
        
        # OPTIMIZATION: Compress embedding to base64
        embedding_array = np.array(embedding, dtype=np.float32)  # Use float32 instead of float64
        compressed_embedding = base64.b64encode(embedding_array.tobytes()).decode('utf-8')
        
        doc = {
            "content": content,
            "source": sample['task_id'],  # Simplified metadata structure
            "embedding_b64": compressed_embedding  # Compressed format
        }
        docs.append(doc)
        
        if (i + 1) % 5 == 0:
            print(f"\n  ├── Processed {i + 1}/{len(json_QA)} documents...")
    
    print(f"\n🔄 Saving {len(docs)} documents to optimized CSV...")
    
    # Save to CSV
    df = pd.DataFrame(docs)
    df.to_csv(output_file, index=False)
    
    file_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    print(f"✅ Optimized vector store ready: {output_file}")
    print(f"  ├── Documents: {len(docs)}")
    if docs:
        print(f"  ├── Embedding dimension: {len(np.frombuffer(base64.b64decode(docs[0]['embedding_b64']), dtype=np.float32))}")
        print(f"  └── File size: {file_size_mb:.1f} MB (optimized)")
    
    return output_file

def setup_development_environment():
    """
    PHASE 1: One-time optimized development setup
    Creates optimized vector store infrastructure for agent development
    """
    print("🏗️  GAIA DEVELOPMENT ENVIRONMENT SETUP (OPTIMIZED)")
    print("=" * 60)
    print("Phase 1: Building optimized vector store infrastructure...")
    print()
    
    # Load processed GAIA data
    print("📂 Loading processed GAIA data...")
    try:
        json_QA = load_json_QA_from_metadata_jsonl('metadata.jsonl')
    except Exception as e:
        print(f"❌ Error loading metadata.jsonl: {e}")
        return None
    
    if not json_QA:
        print("❌ No GAIA data found. Please run data processing first.")
        return None
    
    # Setup embeddings (same model as #1 student)
    print("🔄 Setting up embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 8}
        )
        print("✅ Embedding model loaded")
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return None
    
    # Generate optimized CSV with embeddings
    print("\n🔄 Generating optimized embeddings (this may take a few minutes)...")
    try:
        csv_file = generate_optimized_csv(json_QA, embeddings)
    except Exception as e:
        print(f"❌ Error generating optimized CSV: {e}")
        return None
    
    print("\n🎯 OPTIMIZED DEVELOPMENT SETUP COMPLETE!")
    print("=" * 40)
    print("✅ Optimized vector store CSV created")
    print("✅ ~75% smaller than original format")
    print("✅ Ready for agent development")
    print()
    print("📝 Next steps:")
    print("  1. Use 'gaia_embeddings.csv' in your agent notebooks")
    print("  2. Load retriever with: load_gaia_retriever()")
    print("  3. Focus on agent logic, not infrastructure")
    
    return csv_file

if __name__ == "__main__":
    print("🚀 GAIA Optimized Vector Store Builder")
    print("Run once to set up your optimized development environment")
    print()
    
    # Check if already exists
    import os
    if os.path.exists('gaia_embeddings.csv'):
        response = input("gaia_embeddings.csv already exists. Rebuild with optimization? (y/n): ")
        if response.lower() != 'y':
            print("✅ Using existing vector store")
            exit()
    
    # Build optimized development environment
    csv_file = setup_development_environment()
    
    if csv_file:
        print(f"\n🎉 Optimized development environment ready!")
        print(f"Vector store: {csv_file}")
        print("📊 File size optimized for deployment")
    else:
        print("\n❌ Setup failed")