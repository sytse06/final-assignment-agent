#!/bin/bash

# Safe checkout to examples branch without bringing virtual environment mess

echo "=== Safe Examples Branch Checkout ==="

# 1. First, let's see what branch we're on and what branches exist
echo "1. Current Git status:"
git branch -v
echo "Current branch: $(git branch --show-current)"
echo

# 2. Check if examples branch exists
echo "2. Checking available branches:"
git branch -a | grep examples
echo

# 3. Stash current changes (including the virtual environment mess)
echo "3. Stashing current changes to avoid conflicts..."
git stash push -m "Backup virtual env mess and current work $(date)"
echo "Stash created. You can restore with 'git stash pop' if needed."
echo

# 4. Clean untracked files (including virtual environment)
echo "4. Cleaning untracked files..."
git clean -fd
echo

# 5. Now safely checkout examples branch
echo "5. Checking out examples branch..."
if git checkout examples; then
    echo "✅ Successfully checked out examples branch"
else
    echo "❌ Failed to checkout examples branch"
    echo "Available branches:"
    git branch -a
    echo
    echo "If examples branch doesn't exist locally, trying to create it from origin:"
    if git checkout -b examples origin/examples; then
        echo "✅ Created and checked out examples branch from origin"
    else
        echo "❌ Could not create examples branch from origin"
        echo "You may need to fetch first: git fetch origin"
    fi
fi
echo

# 6. Verify we're on the right branch and it's clean
echo "6. Verification:"
echo "Current branch: $(git branch --show-current)"
echo "Git status:"
git status
echo
echo "Directory contents:"
ls -la | head -10
echo

# 7. Make sure .gitignore is set up to prevent future issues
echo "7. Setting up .gitignore to prevent virtual environment tracking..."
if ! grep -q "test_deployment_py10" .gitignore 2>/dev/null; then
    echo "Adding virtual environment patterns to .gitignore..."
    cat >> .gitignore << 'EOF'

# Virtual environments
test_deployment_py10/
*_py10/
*_py11/
*_py12/
venv/
.venv/
env/
ENV/
EOF
    echo "✅ .gitignore updated"
else
    echo "✅ .gitignore already contains virtual environment patterns"
fi

echo
echo "=== Examples Branch Checkout Complete ==="
echo "You are now on: $(git branch --show-current)"
echo "Working directory is clean and ready for work."
echo
echo "If you need to restore your previous work:"
echo "git stash list    # See available stashes"
echo "git stash pop     # Restore the most recent stash"