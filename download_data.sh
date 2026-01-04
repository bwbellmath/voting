#!/bin/bash
# Download large CSV data files from GitHub Releases

set -e

echo "Downloading large CSV data files from GitHub Releases..."
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Install it with: brew install gh"
    echo ""
    echo "Or download files manually from:"
    echo "https://github.com/bwbellmath/voting/releases/tag/v1.0-data"
    exit 1
fi

# Create directory if it doesn't exist
mkdir -p csv/old

# Download the release files
echo "Downloading files to csv/old/..."
gh release download v1.0-data -D csv/old/ -R bwbellmath/voting --clobber

echo ""
echo "✓ Download complete!"
echo ""
echo "Files downloaded:"
ls -lh csv/old/*.csv | awk '{print "  - " $9 " (" $5 ")"}'
