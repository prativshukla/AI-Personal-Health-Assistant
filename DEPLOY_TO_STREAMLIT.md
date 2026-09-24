# Deploy to Streamlit Community Cloud

This project is prepared for Streamlit Community Cloud.

## 1. Create a GitHub repository

Create an empty repository named, for example:

`HealthAI`

## 2. Upload with Git + Git LFS (recommended)

The two `.pth` model files are about 43 MB each. GitHub blocks individual files over 100 MiB, and Git LFS is the recommended way to track large binary files.

On Mac:

```bash
cd /path/to/HealthAI
brew install git-lfs
git lfs install
git init
git add .
git commit -m "Prepare HealthAI for Streamlit Cloud"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/HealthAI.git
git push -u origin main
```

Because `.gitattributes` already tracks `*.pth`, Git LFS will handle both model files.

## 3. Deploy

Open Streamlit Community Cloud and choose **Create app**.

Select:

- Repository: `YOUR_USERNAME/HealthAI`
- Branch: `main`
- Main file path: `app.py`

Then click **Deploy**.

## 4. Important

Do not remove these files from the repository:

- `pneumonia_model.pth`
- `skin_cancer_model.pth`
- `dataset.csv`
- `symptom_Description.csv`
- `symptom_precaution.csv`
- `Symptom-severity.csv`
- `requirements.txt`
- `app.py`

The app loads all model/data paths relative to `app.py`, so it works both locally and on Streamlit Community Cloud.
