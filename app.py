from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================================================
# AI Personal Health Assistant
# Local, self-contained Streamlit application
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="AI Personal Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f4fbff 0%, #eef7f4 100%);
    }
    [data-testid="stSidebar"] {
        background: #e9f6f3;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #00796b, #0288d1);
        color: white;
        margin-bottom: 1.2rem;
    }
    .warning-box {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #fff8e1;
        border: 1px solid #ffe082;
        color: #5d4b00;
    }
    .result-box {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: white;
        border: 1px solid #dfe8ee;
        box-shadow: 0 2px 12px rgba(0,0,0,.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data helpers
# -----------------------------
@st.cache_data

def load_health_data():
    required = ["dataset.csv", "symptom_Description.csv", "symptom_precaution.csv"]
    missing = [name for name in required if not (BASE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError("Missing project files: " + ", ".join(missing))

    df = pd.read_csv(BASE_DIR / "dataset.csv")
    desc = pd.read_csv(BASE_DIR / "symptom_Description.csv")
    prec = pd.read_csv(BASE_DIR / "symptom_precaution.csv")

    df.columns = df.columns.str.strip()
    desc.columns = desc.columns.str.strip()
    prec.columns = prec.columns.str.strip()

    symptom_cols = [c for c in df.columns if "Symptom" in c]
    if "Disease" not in df.columns or not symptom_cols:
        raise ValueError("dataset.csv does not contain the expected Disease/Symptom columns.")

    clean_symptoms = []
    for col in symptom_cols:
        values = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(" ", "_", regex=False)
        )
        clean_symptoms.extend(values.tolist())

    symptoms = sorted({s for s in clean_symptoms if s and s.lower() != "nan"})

    df["All_Symptoms"] = df[symptom_cols].fillna("").astype(str).apply(
        lambda row: " ".join(
            s.strip().replace(" ", "_")
            for s in row
            if s.strip() and s.strip().lower() != "nan"
        ),
        axis=1,
    )

    return df, desc, prec, symptoms


@st.cache_resource

def train_symptom_model(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["All_Symptoms"])
    model = LogisticRegression(max_iter=1000)
    model.fit(X, df["Disease"])
    return vectorizer, model


# -----------------------------
# Torch model helpers
# -----------------------------
def _build_resnet(num_classes: int):
    # weights=None is intentional: the supplied .pth contains the complete
    # trained ResNet state dict, so the app must not download anything.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@st.cache_resource

def load_xray_model():
    model_path = BASE_DIR / "pneumonia_model.pth"
    if not model_path.exists():
        raise FileNotFoundError("pneumonia_model.pth not found")
    model = _build_resnet(2)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model


@st.cache_resource

def load_skin_model():
    model_path = BASE_DIR / "skin_cancer_model.pth"
    if not model_path.exists():
        raise FileNotFoundError("skin_cancer_model.pth not found")
    model = _build_resnet(7)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def predict_image(model, image):
    image = image.convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted = int(torch.argmax(probabilities).item())
    return predicted, float(probabilities[predicted].item())


# -----------------------------
# Load core data
# -----------------------------
try:
    df, desc, prec, available_symptoms = load_health_data()
    vectorizer, symptom_model = train_symptom_model(df)
except Exception as exc:
    st.error(f"Application setup error: {exc}")
    st.stop()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🩺 Health AI")
    module = st.radio(
        "Navigation",
        ["Introduction", "Symptoms Checker", "Chest X-ray", "Skin Lesion"],
    )
    st.divider()
    st.caption(f"Running on: {DEVICE.type.upper()}")
    st.caption("Models and datasets are loaded locally.")


# -----------------------------
# Introduction
# -----------------------------
if module == "Introduction":
    st.markdown(
        """
        <div class="hero">
            <h1>🧠 AI Personal Health Assistant</h1>
            <p>Three local AI modules for educational health-information support.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🩹 Symptoms")
        st.write("Select symptoms and get a dataset-based disease prediction with description and precautions.")
    with col2:
        st.subheader("🫁 Chest X-ray")
        st.write("Upload a chest X-ray and run the supplied pneumonia classification model.")
    with col3:
        st.subheader("🔬 Skin Lesion")
        st.write("Upload a skin-lesion image and run the supplied 7-class model.")

    image_path = BASE_DIR / "ChatGPT Image Aug 28, 2025, 02_40_38 AM.png"
    if image_path.exists():
        st.image(str(image_path), caption="AI Personal Health Assistant", use_container_width=True)

    st.markdown(
        '<div class="warning-box"><b>Important:</b> This project is for educational/demo purposes. '
        'Predictions are not a medical diagnosis. Do not use the app to delay or replace professional medical care.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------
# Symptoms Checker
# -----------------------------
elif module == "Symptoms Checker":
    st.header("🩹 Symptoms → Disease Prediction")
    st.write("Choose the symptoms that best match what you are experiencing.")

    selected = st.multiselect(
        "Select symptoms",
        available_symptoms,
        placeholder="Start typing a symptom...",
    )

    if st.button("Predict", type="primary", disabled=not selected):
        input_text = " ".join(s.replace(" ", "_") for s in selected)
        input_tfidf = vectorizer.transform([input_text])
        probabilities = symptom_model.predict_proba(input_tfidf)[0]
        best_index = int(probabilities.argmax())
        disease = symptom_model.classes_[best_index]
        confidence = float(probabilities[best_index])

        description_values = desc.loc[desc["Disease"] == disease, "Description"].values
        description = description_values[0] if len(description_values) else "No description available."

        precaution_row = prec[prec["Disease"] == disease]
        precautions = (
            precaution_row.iloc[0, 1:].dropna().tolist()
            if not precaution_row.empty
            else []
        )

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"Predicted condition: {disease}")
        st.write(f"Model confidence: **{confidence:.1%}**")
        st.write("**Description**")
        st.write(description)
        st.write("**Precautions from the dataset**")
        if precautions:
            for item in precautions:
                st.write(f"• {item}")
        else:
            st.write("No precaution listed.")
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Chest X-ray
# -----------------------------
elif module == "Chest X-ray":
    st.header("🫁 Chest X-ray → Pneumonia Prediction")
    st.info("Upload a chest X-ray image. The supplied local ResNet18 model will classify it as NORMAL or PNEUMONIA.")

    uploaded_file = st.file_uploader(
        "Upload chest X-ray",
        type=["jpg", "jpeg", "png"],
        key="xray_upload",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded chest X-ray", use_container_width=True)

        if st.button("Predict Pneumonia", type="primary"):
            try:
                model = load_xray_model()
                index, confidence = predict_image(model, image)
                label = ["NORMAL", "PNEUMONIA"][index]
                if label == "PNEUMONIA":
                    st.warning(f"Prediction: **{label}** ({confidence:.1%} model confidence)")
                else:
                    st.success(f"Prediction: **{label}** ({confidence:.1%} model confidence)")
                st.caption("This is an AI classification result, not a medical diagnosis.")
            except Exception as exc:
                st.error(f"Could not run the X-ray model: {exc}")


# -----------------------------
# Skin Lesion
# -----------------------------
elif module == "Skin Lesion":
    st.header("🔬 Skin Lesion → Classification")
    st.info("Upload a skin-lesion image. The supplied local ResNet18 model predicts one of seven HAM10000-style classes.")

    skin_classes = ["bkl", "nv", "df", "mel", "vasc", "bcc", "akiec"]
    skin_class_names = {
        "bkl": "Benign keratosis-like lesions",
        "nv": "Melanocytic nevi",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "vasc": "Vascular lesions",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses / intraepithelial carcinoma",
    }

    uploaded_file = st.file_uploader(
        "Upload skin-lesion image",
        type=["jpg", "jpeg", "png"],
        key="skin_upload",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded skin-lesion image", use_container_width=True)

        if st.button("Classify Skin Lesion", type="primary"):
            try:
                model = load_skin_model()
                index, confidence = predict_image(model, image)
                code = skin_classes[index]
                label = skin_class_names[code]
                st.warning(f"Predicted class: **{label}** ({confidence:.1%} model confidence)")
                st.caption("This is an AI image-classification result, not a medical diagnosis. Please consult a qualified clinician for evaluation.")
            except Exception as exc:
                st.error(f"Could not run the skin model: {exc}")
