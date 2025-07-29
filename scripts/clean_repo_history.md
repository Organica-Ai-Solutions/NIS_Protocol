# Repository History Cleanup Instructions

These instructions explain how to clean up the repository history to remove large model files that were accidentally committed.

## Prerequisites

1. Download the BFG Repo-Cleaner from: https://rtyley.github.io/bfg-repo-cleaner/
2. Java Runtime Environment (JRE) must be installed

## Backup First!

Always create a backup of your repository before attempting this cleanup:

```bash
# Create a backup
cp -r NIS_Protocol NIS_Protocol_backup
```

## Steps to Clean Repository History

1. Clone a fresh copy of your repository (mirror clone):

```bash
git clone --mirror https://github.com/Organica-Ai-Solutions/NIS_Protocol.git NIS_Protocol.git
```

2. Run BFG to remove large files:

```bash
# Navigate to the directory where you downloaded BFG
cd path/to/bfg

# Remove files larger than 100MB
java -jar bfg.jar --strip-blobs-bigger-than 100M NIS_Protocol.git

# OR remove specific file types
java -jar bfg.jar --delete-files "*.{safetensors,bin,pt,ckpt,model}" NIS_Protocol.git
```

3. Clean up and optimize the repository:

```bash
cd NIS_Protocol.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

4. Push the changes back to GitHub:

```bash
git push
```

## Important Notes

- This process rewrites history, so all collaborators will need to re-clone the repository
- Any pull requests that were open before this operation may need to be re-submitted
- Make sure all team members are aware of this change before proceeding

## Alternative: Using git-filter-repo

If BFG doesn't work for your needs, you can use git-filter-repo instead:

```bash
# Install git-filter-repo
pip install git-filter-repo

# Clone a fresh repository
git clone --mirror https://github.com/Organica-Ai-Solutions/NIS_Protocol.git NIS_Protocol.git
cd NIS_Protocol.git

# Remove large files
git filter-repo --strip-blobs-bigger-than 100M

# Push changes
git push
```

## After Cleanup

After cleaning up the repository:

1. Make sure all team members re-clone the repository
2. Update the .gitignore and .gitattributes files to prevent future issues
3. Set up Git LFS for large files going forward 