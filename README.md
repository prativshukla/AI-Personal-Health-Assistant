# AI Personal Health Assistant

A local Streamlit demo containing:

- Symptoms → disease prediction using TF-IDF + Logistic Regression
- Chest X-ray → pneumonia classification using the supplied ResNet18 model
- Skin lesion → 7-class image classification using the supplied ResNet18 model

## macOS: one-command run

From Terminal, inside this folder:

```bash
./run.sh
```

The script automatically:

1. Finds Python 3.11/3.12/3 if available.
2. Creates `.venv` on the first run.
3. Installs the required packages.
4. Starts Streamlit.
5. Opens the app at the local Streamlit address shown in Terminal.

You can also double-click `run.command` in Finder (macOS may ask for permission the first time).

## Requirements

- macOS
- Python 3.11 or 3.12 recommended
- Internet connection on the **first run only** to install Python packages
- About 1 GB+ free disk space for the Python/ML environment

The trained `.pth` model files are already included, so the app does **not** download ImageNet weights at startup.

## Important medical disclaimer

This project is for educational/demo purposes only. Its outputs are AI/model predictions and are not medical diagnoses. Do not use the application to delay professional medical care.
