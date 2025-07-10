#!/usr/bin/env python3
"""
Debug HuggingFace authentication issues
"""

import os
import requests

def debug_hf_authentication():
    """Debug HF authentication step by step"""
    print("🔍 DEBUGGING HF AUTHENTICATION")
    print("="*50)
    
    # Step 1: Check environment variable
    hf_key = os.getenv('HF_API_KEY')
    print(f"1. Environment variable:")
    if hf_key:
        print(f"   ✅ HF_API_KEY found: {hf_key[:10]}***{hf_key[-8:]}")
        print(f"   Length: {len(hf_key)} characters")
        print(f"   Starts with 'hf_': {hf_key.startswith('hf_')}")
    else:
        print(f"   ❌ HF_API_KEY not found")
        return False
    
    # Step 2: Test different authentication endpoints
    endpoints = [
        "https://huggingface.co/api/whoami",
        "https://api.huggingface.co/api/whoami",
        "https://huggingface.co/api/token",
    ]
    
    headers = {
        'Authorization': f'Bearer {hf_key}',
        'User-Agent': 'HuggingFace-Test/1.0'
    }
    
    print(f"\n2. Testing authentication endpoints:")
    for endpoint in endpoints:
        try:
            print(f"\n   Testing: {endpoint}")
            response = requests.get(endpoint, headers=headers, timeout=10)
            print(f"   Status: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ SUCCESS: {data}")
                    return True
                except:
                    print(f"   ✅ SUCCESS: {response.text[:200]}")
                    return True
            else:
                print(f"   ❌ Failed: {response.text[:200]}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Step 3: Test alternative header formats
    print(f"\n3. Testing alternative header formats:")
    alt_headers = [
        {'Authorization': f'token {hf_key}'},
        {'Authorization': f'hf_{hf_key}' if not hf_key.startswith('hf_') else f'{hf_key}'},
        {'X-API-Key': hf_key},
        {'Authorization': f'Bearer {hf_key}', 'Content-Type': 'application/json'},
    ]
    
    test_url = "https://huggingface.co/api/whoami"
    for i, header in enumerate(alt_headers, 1):
        try:
            print(f"\n   Format {i}: {header}")
            response = requests.get(test_url, headers=header, timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS with format {i}")
                try:
                    data = response.json()
                    print(f"   Response: {data}")
                except:
                    print(f"   Response: {response.text[:100]}")
                return True
            else:
                print(f"   ❌ Failed: {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Step 4: Test without authentication (to check if it's a network issue)
    print(f"\n4. Testing public endpoint (no auth):")
    try:
        public_url = "https://huggingface.co/api/models/bert-base-uncased"
        response = requests.get(public_url, timeout=10)
        print(f"   Public API status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Network connectivity OK")
        else:
            print(f"   ⚠️ Network issue: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Network error: {e}")
    
    return False

def test_key_validity():
    """Test if the key format is valid"""
    print(f"\n🔑 TESTING KEY VALIDITY")
    print("="*50)
    
    hf_key = os.getenv('HF_API_KEY')
    if not hf_key:
        print("❌ No key to test")
        return False
    
    # Check key format
    print(f"Key analysis:")
    print(f"   Full key: {hf_key}")
    print(f"   Length: {len(hf_key)}")
    print(f"   Starts with 'hf_': {hf_key.startswith('hf_')}")
    print(f"   Contains only valid chars: {hf_key.replace('hf_', '').replace('_', '').isalnum()}")
    
    # Expected format: hf_[40 alphanumeric characters]
    if hf_key.startswith('hf_') and len(hf_key) == 43:
        print(f"   ✅ Key format looks correct")
        return True
    else:
        print(f"   ⚠️ Key format might be incorrect")
        print(f"   Expected: hf_ + 40 characters = 43 total")
        print(f"   Actual: {len(hf_key)} characters")
        return False

def test_simple_hf_request():
    """Test a simple request to verify basic access"""
    print(f"\n🌐 TESTING SIMPLE HF REQUEST")
    print("="*50)
    
    hf_key = os.getenv('HF_API_KEY')
    
    # Try the simplest possible authenticated request
    try:
        url = "https://huggingface.co/api/whoami"
        headers = {'Authorization': f'Bearer {hf_key}'}
        
        print(f"URL: {url}")
        print(f"Headers: {headers}")
        
        response = requests.get(url, headers=headers, timeout=15)
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response body: {response.text}")
        
        if response.status_code == 401:
            print(f"\n💡 DIAGNOSIS: 401 Unauthorized")
            print(f"   This usually means:")
            print(f"   1. Invalid API key")
            print(f"   2. API key doesn't have required permissions")
            print(f"   3. API key is expired or revoked")
            print(f"   4. Wrong authentication format")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 HF AUTHENTICATION DEBUG")
    print("="*60)
    
    # Run all debug steps
    has_key = test_key_validity()
    auth_works = debug_hf_authentication()
    simple_works = test_simple_hf_request()
    
    print(f"\n{'='*60}")
    print("📊 DIAGNOSIS SUMMARY:")
    print(f"   🔑 Key format valid: {'✅' if has_key else '❌'}")
    print(f"   🔐 Authentication works: {'✅' if auth_works else '❌'}")
    print(f"   🌐 Simple request works: {'✅' if simple_works else '❌'}")
    
    if not any([has_key, auth_works, simple_works]):
        print(f"\n💡 RECOMMENDED ACTIONS:")
        print(f"   1. Check if your HF API key is valid and active")
        print(f"   2. Try generating a new API key from HF settings")
        print(f"   3. Verify the key has 'read' permissions")
        print(f"   4. Test the key in HF web interface first")
    else:
        print(f"\n✅ Some tests passed - authentication might work with different format")