#!/bin/bash
# Script to set up the NIS Protocol Git repository

# Setup proper directory structure for NIS Protocol

# Create necessary directories if they don't exist
mkdir -p src/core
mkdir -p src/agents
mkdir -p src/emotion
mkdir -p src/memory
mkdir -p src/communication
mkdir -p docs
mkdir -p examples
mkdir -p diagrams
mkdir -p architecture
mkdir -p research

# Move any duplicate directories content to the proper location
# Move docs/emotional_state to emotional_state if not empty
if [ -d "docs/emotional_state" ] && [ "$(ls -A docs/emotional_state)" ]; then
  mkdir -p docs/emotional_state_system
  cp -r docs/emotional_state/* docs/emotional_state_system/
  rm -rf docs/emotional_state
fi

# Move HTML files into docs
if [ -f "emotional_state.html" ]; then
  mv emotional_state.html docs/
fi

if [ -f "architecture.html" ]; then
  mv architecture.html docs/
fi

# Move duplicated assets to proper location
if [ -d "docs/assets" ] && [ "$(ls -A docs/assets)" ]; then
  mkdir -p assets
  cp -r docs/assets/* assets/
  rm -rf docs/assets
fi

# Create necessary files if they don't exist
touch docs/API_Reference.md
touch docs/Implementation_Guide.md
touch docs/Quick_Start_Guide.md

# Initialize Git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit of NIS Protocol"

# Add GitHub remote
git remote add origin git@github.com:Organica-Ai-Solutions/NIS_Protocol.git

echo "Repository structure has been set up correctly!"
echo "To push to GitHub, run:"
echo "git push -u origin main" 