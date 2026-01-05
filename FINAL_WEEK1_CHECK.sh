#!/bin/bash
echo "🔍 WEEK 1 FINAL VERIFICATION"
echo "============================"

# Check local status
echo "1. Local Repository Status:"
git status

# Check commits
echo -e "\n2. Recent Commits:"
git log --oneline -3

# Check tags
echo -e "\n3. Week 1 Tag:"
git tag -l | grep week1

# Check files
echo -e "\n4. Week 1 Deliverables:"
[ -f "notebooks/exploration/EDA.ipynb" ] && echo "✅ EDA notebook exists" || echo "❌ EDA notebook missing"
[ -f "WEEK1_COMPLETION_REPORT.md" ] && echo "✅ Completion report exists" || echo "❌ Completion report missing"
[ -f "README.md" ] && echo "✅ README updated" || echo "❌ README not updated"

# Check synchronization
echo -e "\n5. Synchronization Status:"
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "No remote")
BASE=$(git merge-base @ @{u} 2>/dev/null || echo "No remote")

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "✅ Fully synchronized with remote"
elif [ "$LOCAL" = "$BASE" ]; then
    echo "⚠️  Behind remote - need to pull"
elif [ "$REMOTE" = "$BASE" ]; then
    echo "⚠️  Ahead of remote - need to push"
else
    echo "⚠️  Diverged from remote"
fi

echo -e "\n============================"
echo "🎯 WEEK 1 STATUS: $(if git status | grep -q "nothing to commit"; then echo "COMPLETE ✅"; else echo "IN PROGRESS ⚠️"; fi)"
