#!/bin/bash
set -e

if ! command -v git >/dev/null 2>&1; then
  echo "Git is required. Install Xcode Command Line Tools or Git first."
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Git LFS is required for the .pth model files."
  echo "Install it with: brew install git-lfs"
  exit 1
fi

read -p "GitHub repository URL (e.g. https://github.com/user/AI-Personal-Health-Assistant.git): " REPO

git lfs install
git init
git add .
git commit -m "Prepare AI Personal Health Assistant for Streamlit Cloud" || true
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO"
git push -u origin main

echo "Done. Now open Streamlit Community Cloud and deploy app.py from branch main."
