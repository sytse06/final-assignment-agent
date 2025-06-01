# GAIA Dataset Downloader
# Just loads env variables, downloads GAIA, saves metadata in json and json line format

import os
import json
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def download_gaia_metadata():
    """Download GAIA dataset and save metadata.json locally"""
    
    # Get HF token from environment
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found in .env file")
        print("Add to .env: HF_TOKEN=your_token_here")
        return None
    
    print(f"✅ HF_TOKEN loaded: {hf_token[:10]}...")
    
    try:
        # Load GAIA dataset
        print("🔄 Downloading GAIA dataset...")
        dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", token=hf_token, trust_remote_code=True)
        
        validation_data = dataset["validation"]
        test_data = dataset["test"]
        
        print(f"✅ Dataset loaded:")
        print(f"  Validation: {len(validation_data)} examples")
        print(f"  Test: {len(test_data)} examples")
        
        # Convert to regular Python lists/dicts
        validation_list = [dict(item) for item in validation_data]
        test_list = [dict(item) for item in test_data]
        
        # Save both formats
        # 1. Complete JSON file (easier to use)
        metadata = {
            "validation": validation_list,
            "test": test_list,
            "stats": {
                "validation_count": len(validation_list),
                "test_count": len(test_list),
                "total_count": len(validation_list) + len(test_list)
            }
        }
        
        with open("metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # 2. JSONL format (one line per example)
        with open("metadata.jsonl", "w") as f:
            for item in validation_list:
                json.dump(item, f)
                f.write("\n")
            for item in test_list:
                json.dump(item, f)
                f.write("\n")
        
        print(f"✅ Saved metadata.json and metadata.jsonl ({len(validation_list) + len(test_list)} examples)")
        
        # Show sample
        if validation_list:
            sample = validation_list[0]
            print(f"\n📋 Sample item:")
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 80:
                    print(f"  {key}: '{value[:80]}...'")
                else:
                    print(f"  {key}: {value}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nCheck:")
        print("1. HF token is valid")
        print("2. You have access to gaia-benchmark/GAIA dataset")
        print("3. Internet connection is working")
        return None

def load_gaia_metadata():
    """Load metadata.json if it exists, otherwise download it"""
    
    if os.path.exists("metadata.json"):
        print("📁 Loading existing metadata.json...")
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
        
        print(f"✅ Loaded {metadata['stats']['total_count']} examples from local file")
        return metadata
    else:
        print("📁 metadata.json not found, downloading...")
        return download_gaia_metadata()

# Main execution
if __name__ == "__main__":
    print("🚀 GAIA Dataset Downloader")
    
    # Download/load the data
    metadata = load_gaia_metadata()
    
    if metadata:
        validation_examples = metadata["validation"]
        test_examples = metadata["test"]
        
        print(f"\n🎯 Ready to use:")
        print(f"  validation_examples: {len(validation_examples)} items")
        print(f"  test_examples: {len(test_examples)} items")
        print(f"  File: metadata.json")
        
        # Quick analysis
        levels = {}
        file_count = 0
        for item in validation_examples:
            level = item.get("Level", 1)
            levels[level] = levels.get(level, 0) + 1
            if item.get("file_name"):
                file_count += 1
        
        print(f"\n📊 Quick stats:")
        print(f"  Level distribution: {levels}")
        print(f"  Examples with files: {file_count}")
        
    else:
        print("❌ Failed to get GAIA data")

# Usage examples for your notebooks:
"""
# In your notebook, just do:
from gaia_downloader import load_gaia_metadata

metadata = load_gaia_metadata()
validation_examples = metadata["validation"]

# Use the data
for example in validation_examples[:3]:
    print(f"Q: {example['Question']}")
    print(f"A: {example['Final answer']}")
    print(f"Level: {example['Level']}")
    print("---")
"""