# build_vectorstore.py
# DEVELOPMENT PHASE 1: Build vector store infrastructure once
# Run this script once to set up your development environment

import json
import pandas as pd
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

def generate_development_csv(json_QA: List[Dict], embeddings_model, output_file: str = 'gaia_embeddings.csv'):
    """Generate CSV with embeddings for development use"""
    print(f"📝 Building development vector store: {output_file}")
    print("=" * 50)
    print(f"Processing {len(json_QA)} documents...")
    
    docs = []
    for i, sample in enumerate(json_QA):
        print(f"  Processing document {i + 1}/{len(json_QA)}...", end='\r')
        
        # Same format as your Supabase approach
        content = f"Question : {sample['Question']}\n\nFinal answer : {sample['Final answer']}"
        
        # Compute embedding - this might be slow
        try:
            embedding = embeddings_model.embed_query(content)
        except Exception as e:
            print(f"\n❌ Error computing embedding for document {i + 1}: {e}")
            continue
        
        doc = {
            "content": content,
            "metadata": json.dumps({"source": sample['task_id']}),
            "embedding": json.dumps(embedding)
        }
        docs.append(doc)
        
        if (i + 1) % 5 == 0:
            print(f"\n  ├── Processed {i + 1}/{len(json_QA)} documents...")
    
    print(f"\n🔄 Saving {len(docs)} documents to CSV...")
    
    # Save to CSV
    df = pd.DataFrame(docs)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Development vector store ready: {output_file}")
    print(f"  ├── Documents: {len(docs)}")
    if docs:
        print(f"  ├── Embedding dim: {len(json.loads(docs[0]['embedding']))}")
        print(f"  └── File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    return output_file

def setup_development_environment():
    """
    PHASE 1: One-time development setup
    Creates the vector store infrastructure for agent development
    """
    print("🏗️  GAIA DEVELOPMENT ENVIRONMENT SETUP")
    print("=" * 60)
    print("Phase 1: Building vector store infrastructure...")
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
    
    # Setup embeddings (same model as your Supabase approach)
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
    
    # Generate CSV with embeddings
    print("\n🔄 Generating embeddings (this may take a few minutes)...")
    try:
        csv_file = generate_development_csv(json_QA, embeddings)
    except Exception as e:
        print(f"❌ Error generating CSV: {e}")
        return None
    
    print("\n🎯 DEVELOPMENT SETUP COMPLETE!")
    print("=" * 40)
    print("✅ Vector store CSV created")
    print("✅ Ready for agent development")
    print()
    print("📝 Next steps:")
    print("  1. Use 'gaia_embeddings.csv' in your agent notebooks")
    print("  2. Load retriever with: EmbeddedGAIARetriever('gaia_embeddings.csv')")
    print("  3. Focus on agent logic, not infrastructure")
    
    return csv_file

if __name__ == "__main__":
    print("🚀 GAIA Development Vector Store Builder")
    print("Run once to set up your development environment")
    print()
    
    # Check if already exists
    import os
    if os.path.exists('gaia_embeddings.csv'):
        response = input("gaia_embeddings.csv already exists. Rebuild? (y/n): ")
        if response.lower() != 'y':
            print("✅ Using existing vector store")
            exit()
    
    # Build development environment
    csv_file = setup_development_environment()
    
    if csv_file:
        print(f"\n🎉 Development environment ready!")
        print(f"Vector store: {csv_file}")
    else:
        print("\n❌ Setup failed")