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
# AI Personal Health Assistant — polished Streamlit UI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="HealthAI • Personal Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Design system
# -----------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #f6f8fc;
        --surface: #ffffff;
        --surface-2: #f0f7f7;
        --text: #10233f;
        --muted: #64748b;
        --line: #e5eaf1;
        --primary: #0f766e;
        --primary-2: #14b8a6;
        --blue: #2563eb;
        --danger: #dc2626;
        --warning: #d97706;
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #e8fbf7 0%, transparent 28%),
                    radial-gradient(circle at 100% 10%, #eaf1ff 0%, transparent 30%),
                    var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] { background: rgba(246,248,252,.82); }
    [data-testid="stToolbar"] { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f35 0%, #102b47 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #eaf2fb !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 12px;
        padding: 10px 12px;
    }

    .brand {
        display:flex; align-items:center; gap:12px;
        padding: 6px 4px 22px;
    }
    .brand-icon {
        width:44px; height:44px; border-radius:14px;
        display:flex; align-items:center; justify-content:center;
        background: linear-gradient(135deg, #14b8a6, #2563eb);
        font-size:23px;
        box-shadow: 0 10px 25px rgba(20,184,166,.22);
    }
    .brand-title { font-size:18px; font-weight:800; line-height:1.1; }
    .brand-sub { font-size:11px; color:#9fb2c8 !important; margin-top:3px; }

    .side-status {
        margin-top: 22px; padding: 13px 14px; border-radius: 14px;
        background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.09);
        font-size: 12px;
    }
    .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#34d399; margin-right:7px; }

    .page-kicker {
        color: var(--primary); font-size: 12px; font-weight: 800;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: 4px;
    }
    .page-title { font-size: 34px; line-height:1.08; font-weight:850; color:var(--text); margin:0; }
    .page-sub { color:var(--muted); font-size:15px; margin-top:8px; margin-bottom:20px; }

    .hero {
        position:relative; overflow:hidden; padding:30px 32px; border-radius:24px;
        background: linear-gradient(135deg, #0b2942 0%, #0f766e 62%, #159e92 100%);
        color:white; margin: 6px 0 22px;
        box-shadow: 0 18px 45px rgba(15,118,110,.18);
    }
    .hero:after {
        content:""; position:absolute; width:240px; height:240px; border-radius:50%;
        right:-75px; top:-100px; background:rgba(255,255,255,.08);
    }
    .hero h1 { margin:0; font-size:36px; letter-spacing:-.03em; }
    .hero p { margin:10px 0 0; max-width:700px; color:#d7f8f4; font-size:15px; }
    .hero-tag {
        display:inline-flex; margin-bottom:14px; padding:6px 10px; border-radius:999px;
        background:rgba(255,255,255,.13); color:#eafffc; font-size:11px; font-weight:700;
    }

    .metric-card, .feature-card, .panel, .result-card {
        background:rgba(255,255,255,.92); border:1px solid var(--line);
        border-radius:18px; box-shadow:0 8px 28px rgba(15,35,63,.06);
    }
    .metric-card { padding:20px; min-height:142px; }
    .metric-icon { font-size:24px; }
    .metric-title { margin-top:10px; font-weight:800; color:var(--text); font-size:16px; }
    .metric-text { color:var(--muted); font-size:13px; line-height:1.55; margin-top:5px; }

    .feature-card { padding:18px 18px 17px; min-height:150px; }
    .feature-top { display:flex; align-items:center; gap:10px; }
    .feature-icon { width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center; background:#e8f7f5; font-size:19px; }
    .feature-name { font-weight:800; color:var(--text); }
    .feature-card p { color:var(--muted); font-size:13px; line-height:1.55; margin:12px 0 0; }

    .section-label { font-size:12px; font-weight:800; color:#64748b; text-transform:uppercase; letter-spacing:.1em; margin:4px 0 10px; }
    .panel { padding:22px; margin: 8px 0 18px; }

    .upload-panel {
        border:1.5px dashed #b8c8d8; border-radius:18px; padding:18px;
        background:linear-gradient(180deg,#fbfdff,#f6fafc);
    }

    .result-card { padding:22px; margin-top:18px; }
    .result-head { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; }
    .result-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; font-weight:800; }
    .result-value { font-size:26px; font-weight:850; color:var(--text); margin-top:3px; }
    .confidence { font-size:12px; color:var(--muted); margin-top:12px; }
    .bar { height:8px; border-radius:99px; background:#e7edf3; overflow:hidden; margin-top:6px; }
    .bar > div { height:100%; border-radius:99px; background:linear-gradient(90deg,#0f766e,#14b8a6); }

    .mini-note {
        padding:12px 14px; border-radius:12px; background:#f4f8fb; color:#5f7185;
        font-size:12px; line-height:1.5; border:1px solid #e5ebf1;
    }
    .medical-note {
        padding:13px 15px; border-radius:13px; background:#fff8e8;
        border:1px solid #f6d78b; color:#6b4e08; font-size:12px; line-height:1.55;
    }

    .stButton > button {
        border-radius:12px !important; font-weight:750 !important;
        min-height:44px !important; border:0 !important;
        box-shadow:0 7px 18px rgba(15,118,110,.12);
    }
    .stButton > button[kind="primary"] {
        background:linear-gradient(135deg,#0f766e,#14b8a6) !important;
    }
    .stTextInput input, .stTextArea textarea, .stMultiSelect div[data-baseweb="select"] {
        border-radius:12px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        border-radius:14px !important; border:1.5px dashed #b8c8d8 !important;
        background:#fbfdff !important;
    }

    @media (max-width: 800px) {
        .page-title, .hero h1 { font-size:28px; }
        .hero { padding:24px; }
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
            df[col].dropna().astype(str).str.strip().str.replace(" ", "_", regex=False)
        )
        clean_symptoms.extend(values.tolist())
    symptoms = sorted({s for s in clean_symptoms if s and s.lower() != "nan"})

    df["All_Symptoms"] = df[symptom_cols].fillna("").astype(str).apply(
        lambda row: " ".join(
            s.strip().replace(" ", "_") for s in row
            if s.strip() and s.strip().lower() != "nan"
        ), axis=1,
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


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


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
# Sidebar navigation
# -----------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-icon">🩺</div>'
        '<div><div class="brand-title">HealthAI</div>'
        '<div class="brand-sub">Personal Health Assistant</div></div></div>',
        unsafe_allow_html=True,
    )

    st.caption("WORKSPACE")
    module = st.radio(
        "Navigation",
        ["Overview", "Symptoms Checker", "Chest X-ray", "Skin Lesion"],
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="side-status"><span class="dot"></span><b>AI engine ready</b><br>'
        f'<span style="color:#9fb2c8">Device: {DEVICE.type.upper()} · Models loaded on demand</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.caption("Educational / demonstration app")


# -----------------------------
# Overview
# -----------------------------
if module == "Overview":
    st.markdown('<div class="page-kicker">AI-powered health workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Understand your health data, faster.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">A clean dashboard for symptom analysis and image-based health model demos — built for learning and experimentation.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero"><div class="hero-tag">● READY TO EXPLORE</div>'
        '<h1>Your health, one simple workspace.</h1>'
        '<p>Explore three AI-assisted modules: symptom-based prediction, chest X-ray classification, and skin-lesion image classification.</p></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    cards = [
        ("🩹", "Symptoms Checker", "Select symptoms and receive a dataset-based prediction with a description and listed precautions."),
        ("🫁", "Chest X-ray", "Upload a chest X-ray and run the supplied ResNet18 pneumonia classification model."),
        ("🔬", "Skin Lesion", "Upload a skin-lesion image and run the supplied seven-class image classification model."),
    ]
    for col, (icon, title, text) in zip((c1, c2, c3), cards):
        with col:
            st.markdown(
                f'<div class="feature-card"><div class="feature-top"><div class="feature-icon">{icon}</div>'
                f'<div class="feature-name">{title}</div></div><p>{text}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3, gap="medium")
    for col, n, title, text in [
        (h1, "01", "Choose a module", "Use the left navigation to open the workflow you need."),
        (h2, "02", "Provide input", "Select symptoms or upload a supported image file."),
        (h3, "03", "Review output", "Read the model result, confidence and supporting information."),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div style="font-size:12px;font-weight:800;color:#0f766e">{n}</div>'
                f'<div class="metric-title">{title}</div><div class="metric-text">{text}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="medical-note"><b>Important:</b> This project is for educational/demo purposes only. '
        'AI outputs are model predictions, not medical diagnoses. Do not use this application to delay professional medical care.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------
# Symptoms Checker
# -----------------------------
elif module == "Symptoms Checker":
    st.markdown('<div class="page-kicker">Module 01 · Symptom analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Symptoms Checker</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Select the symptoms that best match your current experience, then run the dataset-based model.</div>', unsafe_allow_html=True)

    selected = st.multiselect(
        "Select symptoms",
        available_symptoms,
        placeholder="Search and add symptoms…",
    )

    st.markdown(
        f'<div class="mini-note"><b>{len(selected)} symptoms selected.</b> '
        'For a useful demo result, choose the symptoms that most closely match the input.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Analyze symptoms →", type="primary", use_container_width=True, disabled=not selected):
        with st.spinner("Analyzing selected symptoms…"):
            input_text = " ".join(s.replace(" ", "_") for s in selected)
            input_tfidf = vectorizer.transform([input_text])
            probabilities = symptom_model.predict_proba(input_tfidf)[0]
            best_index = int(probabilities.argmax())
            disease = symptom_model.classes_[best_index]
            confidence = float(probabilities[best_index])

            description_values = desc.loc[desc["Disease"] == disease, "Description"].values
            description = description_values[0] if len(description_values) else "No description available."
            precaution_row = prec[prec["Disease"] == disease]
            precautions = precaution_row.iloc[0, 1:].dropna().tolist() if not precaution_row.empty else []

        st.markdown(
            f'<div class="result-card"><div class="result-head"><div>'
            f'<div class="result-label">Model result</div><div class="result-value">{disease}</div>'
            f'</div><div style="font-size:30px">🧠</div></div>'
            f'<div class="confidence">Model confidence · {confidence:.1%}</div>'
            f'<div class="bar"><div style="width:{max(4, confidence*100):.1f}%"></div></div></div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown('<div class="panel"><div class="section-label">Description</div>', unsafe_allow_html=True)
            st.write(description)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="panel"><div class="section-label">Dataset precautions</div>', unsafe_allow_html=True)
            if precautions:
                for item in precautions:
                    st.markdown(f"- {item}")
            else:
                st.write("No precaution listed in the dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="medical-note"><b>Reminder:</b> This prediction is generated from the project dataset and should not be treated as a diagnosis.</div>',
            unsafe_allow_html=True,
        )


# -----------------------------
# Chest X-ray
# -----------------------------
elif module == "Chest X-ray":
    st.markdown('<div class="page-kicker">Module 02 · Image classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Chest X-ray</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload a chest X-ray to run the supplied ResNet18 pneumonia classification model.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.05], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="section-label">Upload image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG or PNG file",
            type=["jpg", "jpeg", "png"],
            key="xray_upload",
            label_visibility="visible",
        )
        if uploaded_file:
            st.success("Image ready for analysis")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            if st.button("Analyze X-ray →", type="primary", use_container_width=True):
                try:
                    with st.spinner("Running pneumonia model…"):
                        model = load_xray_model()
                        index, confidence = predict_image(model, image)
                    label = ["NORMAL", "PNEUMONIA"][index]
                    tone = "#dc2626" if label == "PNEUMONIA" else "#0f766e"
                    st.markdown(
                        f'<div class="result-card"><div class="result-label">Prediction</div>'
                        f'<div class="result-value" style="color:{tone}">{label}</div>'
                        f'<div class="confidence">Model confidence · {confidence:.1%}</div>'
                        f'<div class="bar"><div style="width:{max(4, confidence*100):.1f}%;background:{tone}"></div></div></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error(f"Could not run the X-ray model: {exc}")
        else:
            st.markdown(
                '<div class="panel" style="min-height:310px;display:flex;align-items:center;justify-content:center;">'
                '<div style="text-align:center;color:#64748b"><div style="font-size:42px">🫁</div>'
                '<div style="font-weight:800;color:#10233f;margin-top:8px">Your preview will appear here</div>'
                '<div style="font-size:13px;margin-top:4px">Upload a chest X-ray to continue.</div></div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="medical-note"><b>Medical safety:</b> This is an AI image-classification demo, not a medical diagnosis. '
        'A qualified healthcare professional should interpret chest X-rays.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------
# Skin Lesion
# -----------------------------
elif module == "Skin Lesion":
    st.markdown('<div class="page-kicker">Module 03 · Image classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Skin Lesion</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload a skin-lesion image to run the supplied seven-class ResNet18 model.</div>', unsafe_allow_html=True)

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

    left, right = st.columns([1, 1.05], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="section-label">Upload image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG or PNG file",
            type=["jpg", "jpeg", "png"],
            key="skin_upload",
        )
        if uploaded_file:
            st.success("Image ready for analysis")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            if st.button("Classify lesion →", type="primary", use_container_width=True):
                try:
                    with st.spinner("Running skin-lesion model…"):
                        model = load_skin_model()
                        index, confidence = predict_image(model, image)
                    code = skin_classes[index]
                    label = skin_class_names[code]
                    st.markdown(
                        f'<div class="result-card"><div class="result-label">Predicted class</div>'
                        f'<div class="result-value">{label}</div>'
                        f'<div class="confidence">Model confidence · {confidence:.1%}</div>'
                        f'<div class="bar"><div style="width:{max(4, confidence*100):.1f}%"></div></div></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error(f"Could not run the skin model: {exc}")
        else:
            st.markdown(
                '<div class="panel" style="min-height:310px;display:flex;align-items:center;justify-content:center;">'
                '<div style="text-align:center;color:#64748b"><div style="font-size:42px">🔬</div>'
                '<div style="font-weight:800;color:#10233f;margin-top:8px">Your preview will appear here</div>'
                '<div style="font-size:13px;margin-top:4px">Upload a skin image to continue.</div></div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="medical-note"><b>Medical safety:</b> This is an AI image-classification demo, not a medical diagnosis. '
        'Skin images should be evaluated by a qualified clinician when medical concern exists.</div>',
        unsafe_allow_html=True,
    )
