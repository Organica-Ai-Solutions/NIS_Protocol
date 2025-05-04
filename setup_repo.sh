#!/bin/bash
# Script to set up the NIS Protocol Git repository

# Initialize Git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit of NIS Protocol"

# Add GitHub remote
git remote add origin git@github.com:Organica-Ai-Solutions/NIS_Protocol.git

echo "Repository setup complete!"
echo "To push to GitHub, run:"
echo "git push -u origin main" 