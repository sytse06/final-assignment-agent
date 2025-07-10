#!/usr/bin/env python3
"""
Test file access directly without authentication first
This will show us if files are publicly accessible
"""

import requests
from pathlib import Path

def test_direct_file_access():
    """Test if GAIA files are publicly accessible"""
    print("🌐 TESTING DIRECT FILE ACCESS (NO AUTH)")
    print("="*60)
    
    # Your specific tower file
    file_name = "389793a7-ca17-4e82-81cb-2b3a2391b4b9.txt"
    
    # Try multiple URL patterns without authentication
    url_patterns = [
        f"https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/{file_name}",
        f"https://huggingface.co/datasets/gaia-benchmark/GAIA/raw/main/2023/validation/{file_name}",
        f"https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/validation/{file_name}",
        f"https://huggingface.co/datasets/gaia-benchmark/GAIA/raw/main/validation/{file_name}",
        # Try without subdirectories
        f"https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/{file_name}",
        f"https://huggingface.co/datasets/gaia-benchmark/GAIA/raw/main/{file_name}",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/plain, text/html, application/octet-stream, */*',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    print(f"📝 Testing file: {file_name}")
    print(f"🔍 Trying {len(url_patterns)} URL patterns...")
    
    for i, url in enumerate(url_patterns, 1):
        print(f"\n📡 Pattern {i}: {url}")
        
        try:
            # Test with HEAD first
            response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
            print(f"   HEAD Status: {response.status_code}")
            
            if response.status_code == 200:
                # Try GET
                response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
                if response.status_code == 200:
                    content = response.text
                    print(f"   ✅ SUCCESS! Downloaded {len(content)} characters")
                    print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
                    
                    print(f"\n📄 TOWER LAYOUT CONTENT:")
                    print("="*50)
                    print(content)
                    print("="*50)
                    
                    # Analyze
                    lines = content.strip().split('\n')
                    h_count = content.count('H')
                    dash_count = content.count('-')
                    
                    print(f"\n📊 Analysis:")
                    print(f"   Lines: {len(lines)}")
                    print(f"   Houses (H): {h_count}")
                    print(f"   Mile markers (-): {dash_count}")
                    
                    print(f"\n🎯 PERFECT! This is exactly what your agent needs!")
                    return url, content
                else:
                    print(f"   GET failed: {response.status_code}")
            elif response.status_code == 404:
                print(f"   ❌ Not found (404)")
            elif response.status_code == 401:
                print(f"   🔐 Requires authentication (401)")
            elif response.status_code == 403:
                print(f"   🚫 Forbidden (403)")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n⚠️ No patterns worked without authentication")
    return None, None

def test_other_gaia_files():
    """Test a few other known GAIA files"""
    print(f"\n📚 TESTING OTHER GAIA FILES")
    print("="*50)
    
    # Some other GAIA files to test
    test_files = [
        "32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx",  # From your earlier test
        "test.txt",  # Generic test
        "README.md",  # Might exist
    ]
    
    base_urls = [
        "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation",
        "https://huggingface.co/datasets/gaia-benchmark/GAIA/raw/main/2023/validation",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    for file_name in test_files:
        print(f"\n📄 Testing: {file_name}")
        
        for base_url in base_urls:
            url = f"{base_url}/{file_name}"
            try:
                response = requests.head(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    print(f"   ✅ Found at: {url}")
                    break
                elif response.status_code == 401:
                    print(f"   🔐 Auth required: {url}")
                    break
            except:
                pass
        else:
            print(f"   ❌ Not found in any location")

def test_dataset_info():
    """Test if we can get dataset info without auth"""
    print(f"\n📊 TESTING DATASET INFO")
    print("="*40)
    
    info_urls = [
        "https://huggingface.co/datasets/gaia-benchmark/GAIA",
        "https://huggingface.co/api/datasets/gaia-benchmark/GAIA",
        "https://huggingface.co/datasets/gaia-benchmark/GAIA/tree/main",
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for url in info_urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"{url}: {response.status_code}")
            
            if response.status_code == 200 and 'gaia' in response.text.lower():
                print(f"   ✅ Dataset info accessible")
            
        except Exception as e:
            print(f"{url}: Error - {e}")

if __name__ == "__main__":
    print("🌐 DIRECT FILE ACCESS TEST")
    print("Testing GAIA files without authentication")
    print("="*70)
    
    # Test your specific file
    success_url, content = test_direct_file_access()
    
    # Test other files
    test_other_gaia_files()
    
    # Test dataset info
    test_dataset_info()
    
    print(f"\n{'='*70}")
    print("📊 RESULTS:")
    
    if success_url:
        print(f"✅ SUCCESS: Files are publicly accessible!")
        print(f"🔗 Working URL pattern: {success_url}")
        print(f"🎯 Your ContentRetrieverTool will work in deployment!")
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Update your agent to use this URL pattern")
        print(f"   2. Test ContentRetrieverTool with this URL")
        print(f"   3. Deploy with confidence!")
    else:
        print(f"❌ Files require authentication")
        print(f"💡 NEXT STEPS:")
        print(f"   1. Generate a new valid HF API key")
        print(f"   2. Test authentication again")
        print(f"   3. Files will be accessible in HF Spaces with automatic auth")