#!/usr/bin/env python3
from tools import check_vision_status

if __name__ == "__main__":
    success = check_vision_status()
    if success:
        print("\n🎉 All vision tools ready!")
    else:
        print("\n⚠️  Issues found - check output above")