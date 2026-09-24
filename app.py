import html
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
# HealthAI — fast, stateful Streamlit UI
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="HealthAI | Personal Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Theme / UI
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#f4f7fb;--card:#fff;--ink:#10233f;--muted:#66768a;--line:#e5ebf2;--teal:#0f766e;--teal2:#16b8a6;--navy:#0b1f35;--blue:#2563eb;--danger:#dc2626}
html,body,[class*="css"]{font-family:Inter,system-ui,sans-serif}
.stApp{background:linear-gradient(135deg,#f7fafc 0%,#eff8f7 48%,#f4f7ff 100%);color:var(--ink)}
[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.block-container{max-width:1280px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#081b30,#0d2944 75%,#0b223a);border-right:1px solid rgba(255,255,255,.08)}
[data-testid="stSidebar"] *{color:#edf6ff}
[data-testid="stSidebar"] .stRadio label{border-radius:12px;padding:10px 12px;transition:.18s}
[data-testid="stSidebar"] .stRadio label:hover{background:rgba(255,255,255,.08)}
.brand{display:flex;align-items:center;gap:12px;padding:4px 2px 24px}
.brandIcon{width:46px;height:46px;border-radius:15px;background:linear-gradient(135deg,#16b8a6,#2563eb);display:flex;align-items:center;justify-content:center;font-size:24px;box-shadow:0 12px 30px rgba(20,184,166,.25)}
.brandTitle{font-size:19px;font-weight:800}.brandSub{font-size:11px;color:#9eb4c9!important;margin-top:2px}
.sideBox{margin-top:18px;padding:13px;border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.06);border-radius:14px;font-size:12px;line-height:1.55}.greenDot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#34d399;margin-right:7px}
.eyebrow{font-size:11px;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:var(--teal);margin-bottom:7px}.title{font-size:38px;font-weight:800;line-height:1.08;letter-spacing:-.035em}.subtitle{font-size:15px;color:var(--muted);max-width:760px;line-height:1.6;margin-top:9px;margin-bottom:22px}
.hero{position:relative;overflow:hidden;border-radius:26px;padding:34px;margin:8px 0 22px;background:linear-gradient(125deg,#09243c,#0f766e 68%,#18a999);color:#fff;box-shadow:0 20px 50px rgba(15,118,110,.17)}
.hero:before{content:"";position:absolute;width:300px;height:300px;border-radius:50%;right:-110px;top:-150px;background:rgba(255,255,255,.08)}.hero:after{content:"";position:absolute;width:130px;height:130px;border-radius:50%;right:120px;bottom:-90px;background:rgba(255,255,255,.06)}
.hero>*{position:relative;z-index:1}.heroBadge{display:inline-flex;padding:6px 10px;border-radius:99px;background:rgba(255,255,255,.13);font-size:11px;font-weight:700}.hero h1{font-size:36px;line-height:1.08;margin:13px 0 8px;letter-spacing:-.03em}.hero p{margin:0;color:#d9faf6;max-width:730px;line-height:1.6;font-size:14px}
.card{background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:19px;box-shadow:0 10px 32px rgba(16,35,63,.055)}
.feature{padding:20px;min-height:175px}.featureIcon{width:42px;height:42px;border-radius:13px;background:#e7f7f4;display:flex;align-items:center;justify-content:center;font-size:21px;margin-bottom:13px}.featureTitle{font-weight:800;font-size:16px}.featureText{font-size:13px;color:var(--muted);line-height:1.55;margin-top:7px}.pill{display:inline-block;margin-top:13px;font-size:10px;font-weight:800;color:var(--teal);background:#e9f8f5;padding:5px 8px;border-radius:99px}
.step{padding:18px;min-height:132px}.stepNo{font-size:11px;font-weight:800;color:var(--teal)}.stepTitle{font-size:15px;font-weight:800;margin-top:8px}.stepText{font-size:12px;color:var(--muted);line-height:1.5;margin-top:5px}
.section{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#738398;margin:5px 0 10px}.panel{padding:22px;margin:8px 0 18px}.hint{padding:12px 14px;border-radius:13px;background:#f4f8fb;border:1px solid #e6edf3;color:#607287;font-size:12px;line-height:1.5}.notice{padding:14px 16px;border-radius:14px;background:#fff8e8;border:1px solid #f3d88d;color:#6c5008;font-size:12px;line-height:1.55;margin-top:18px}.successBox{padding:12px 14px;border-radius:12px;background:#e9f8f3;border:1px solid #c6eee3;color:#116b5f;font-size:12px;font-weight:600}
.result{padding:23px;margin-top:18px}.resultLabel{font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:#728297}.resultValue{font-size:28px;font-weight:800;line-height:1.15;margin-top:5px}.confidence{font-size:12px;color:var(--muted);margin-top:13px}.bar{height:8px;border-radius:99px;background:#e8eef4;overflow:hidden;margin-top:7px}.bar>div{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--teal),var(--teal2))}
.stButton>button{min-height:45px!important;border-radius:13px!important;font-weight:750!important;border:1px solid #dce6ee!important;background:#fff!important;color:var(--ink)!important;box-shadow:0 5px 18px rgba(16,35,63,.05)!important;transition:transform .15s,box-shadow .15s!important}.stButton>button:hover{transform:translateY(-1px);box-shadow:0 9px 24px rgba(16,35,63,.10)!important}.stButton>button[kind="primary"]{border:0!important;background:linear-gradient(135deg,#0f766e,#16b8a6)!important;color:white!important}
div[data-baseweb="select"]>div{border-radius:13px!important;border-color:#dce6ee!important;background:#fff!important}.stTextInput input{border-radius:13px!important}.stFileUploader>section{border-radius:15px!important;border:1.5px dashed #b8c9d9!important;background:#fbfdff!important}.stProgress>div>div{background:linear-gradient(90deg,#0f766e,#16b8a6)}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);padding:15px;border-radius:16px}.smallMuted{font-size:12px;color:var(--muted)}
/* --- UI fixes --- */
.hero h1{color:#fff!important}.hero p{color:#d9faf6!important}.hero{color:#fff}
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-secondaryFormSubmit"]{min-height:45px;border-radius:13px;font-weight:700;border:1px solid #dce6ee;background:#fff;color:var(--ink);box-shadow:0 5px 18px rgba(16,35,63,.05)}
[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{min-height:48px;border-radius:13px;font-weight:700;border:0;background:linear-gradient(135deg,#0f766e,#16b8a6);color:#fff;box-shadow:0 8px 22px rgba(15,118,110,.28);transition:transform .15s,filter .15s}
[data-testid="stBaseButton-primary"] p,[data-testid="stBaseButton-primaryFormSubmit"] p{color:#fff!important}
[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-primaryFormSubmit"]:hover{filter:brightness(1.06);transform:translateY(-1px)}
[data-testid="stForm"]{background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:19px;padding:22px;box-shadow:0 10px 32px rgba(16,35,63,.055)}
[data-testid="stSidebar"] .stRadio [role="radiogroup"]{gap:4px}
[data-testid="stSidebar"] .stRadio label{width:100%;border:1px solid transparent;cursor:pointer}
[data-testid="stSidebar"] .stRadio label>div:first-child{display:none}
[data-testid="stSidebar"] .stRadio label:has(input:checked){background:linear-gradient(135deg,rgba(22,184,166,.30),rgba(37,99,235,.24));border-color:rgba(255,255,255,.16);font-weight:700}
[data-testid="stFileUploader"] section{padding:22px}
@media(max-width:800px){[data-testid="stForm"]{padding:14px}.card.result{padding:18px}.resultValue{font-size:23px}}
@media(max-width:800px){.block-container{padding:1rem}.title{font-size:29px}.hero{padding:25px}.hero h1{font-size:29px}.feature{min-height:0}}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# State helpers
# -----------------------------
if "module" not in st.session_state:
    st.session_state.module = "Overview"
if "symptom_result" not in st.session_state:
    st.session_state.symptom_result = None
if "xray_result" not in st.session_state:
    st.session_state.xray_result = None
if "skin_result" not in st.session_state:
    st.session_state.skin_result = None

MODULES = ["Overview", "Symptoms Checker", "Chest X-ray", "Skin Lesion"]


def go_to(name: str):
    st.session_state.module = name


# -----------------------------
# Data helpers (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_health_data():
    required = ["dataset.csv", "symptom_Description.csv", "symptom_precaution.csv"]
    missing = [x for x in required if not (BASE_DIR / x).exists()]
    if missing:
        raise FileNotFoundError("Missing project files: " + ", ".join(missing))
    df = pd.read_csv(BASE_DIR / "dataset.csv")
    desc = pd.read_csv(BASE_DIR / "symptom_Description.csv")
    prec = pd.read_csv(BASE_DIR / "symptom_precaution.csv")
    df.columns = df.columns.str.strip(); desc.columns = desc.columns.str.strip(); prec.columns = prec.columns.str.strip()
    symptom_cols = [c for c in df.columns if "Symptom" in c]
    if "Disease" not in df.columns or not symptom_cols:
        raise ValueError("dataset.csv does not contain the expected Disease/Symptom columns.")
    symptoms = []
    for col in symptom_cols:
        vals = df[col].dropna().astype(str).str.strip().str.replace(" ", "_", regex=False).tolist()
        symptoms.extend([x for x in vals if x and x.lower() != "nan"])
    df["All_Symptoms"] = df[symptom_cols].fillna("").astype(str).apply(
        lambda row: " ".join(s.strip().replace(" ", "_") for s in row if s.strip() and s.strip().lower() != "nan"), axis=1
    )
    return df, desc, prec, sorted(set(symptoms))


@st.cache_resource(show_spinner=False)
def train_symptom_model(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["All_Symptoms"])
    model = LogisticRegression(max_iter=1000)
    model.fit(X, df["Disease"])
    return vectorizer, model


def _build_resnet(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@st.cache_resource(show_spinner="Loading X-ray model…")
def load_xray_model():
    path = BASE_DIR / "pneumonia_model.pth"
    if not path.exists():
        raise FileNotFoundError("pneumonia_model.pth not found")
    model = _build_resnet(2)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True); model.to(DEVICE).eval()
    return model


@st.cache_resource(show_spinner="Loading skin model…")
def load_skin_model():
    path = BASE_DIR / "skin_cancer_model.pth"
    if not path.exists():
        raise FileNotFoundError("skin_cancer_model.pth not found")
    model = _build_resnet(7)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True); model.to(DEVICE).eval()
    return model


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def predict_image(model, image):
    tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0]
        index = int(torch.argmax(probs).item())
    return index, float(probs[index].item())


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
    st.markdown('<div class="brand"><div class="brandIcon">🩺</div><div><div class="brandTitle">HealthAI</div><div class="brandSub">Personal Health Assistant</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section" style="color:#9eb4c9">WORKSPACE</div>', unsafe_allow_html=True)
    choice = st.radio("Navigation", MODULES, index=MODULES.index(st.session_state.module), label_visibility="collapsed")
    if choice != st.session_state.module:
        st.session_state.module = choice
        st.rerun()
    st.markdown(f'<div class="sideBox"><span class="greenDot"></span><b>System ready</b><br><span style="color:#9eb4c9">CPU/GPU: {DEVICE.type.upper()} · Data engine cached</span></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.caption("Educational / demonstration app")

module = st.session_state.module

# -----------------------------
# Overview
# -----------------------------
if module == "Overview":
    st.markdown('<div class="eyebrow">AI-powered health workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="title">Your health, one simple workspace.</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore symptom analysis and image-classification demos in one clean, responsive interface.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><span class="heroBadge">● READY TO EXPLORE</span><h1>Understand your health data faster.</h1><p>Use the three project modules to select symptoms, upload a chest X-ray, or upload a skin-lesion image and review the model output.</p></div>', unsafe_allow_html=True)

    cols = st.columns(3, gap="medium")
    features = [
        ("🩹", "Symptoms Checker", "Select symptoms and get a dataset-based prediction, description and listed precautions.", "TEXT MODEL"),
        ("🫁", "Chest X-ray", "Upload an image and run the supplied ResNet18 pneumonia classifier.", "IMAGE MODEL"),
        ("🔬", "Skin Lesion", "Upload an image and run the supplied seven-class ResNet18 classifier.", "IMAGE MODEL"),
    ]
    for col, (icon, title, text, tag) in zip(cols, features):
        with col:
            st.markdown(f'<div class="card feature"><div class="featureIcon">{icon}</div><div class="featureTitle">{title}</div><div class="featureText">{text}</div><span class="pill">{tag}</span></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section">Quick start</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3, gap="medium")
    for col, no, title, text, target in [
        (a,"01","Check symptoms","Choose symptoms and submit once when ready.","Symptoms Checker"),
        (b,"02","Analyze X-ray","Upload an image and press Analyze once.","Chest X-ray"),
        (c,"03","Classify lesion","Upload a supported image and submit.","Skin Lesion"),
    ]:
        with col:
            st.markdown(f'<div class="card step"><div class="stepNo">{no}</div><div class="stepTitle">{title}</div><div class="stepText">{text}</div></div>', unsafe_allow_html=True)
            st.button(f"Open {title} →", key=f"open_{target}", use_container_width=True, on_click=go_to, args=(target,))
    st.markdown('<div class="notice"><b>Important:</b> This project is for educational/demo purposes. Model outputs are predictions and are not a medical diagnosis. Do not use the app to delay professional care.</div>', unsafe_allow_html=True)

# -----------------------------
# Symptoms
# -----------------------------
if module == "Symptoms Checker":
    st.markdown('<div class="eyebrow">Module 01 · Symptom analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="title">Symptoms Checker</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Select the symptoms that best match the project dataset, then submit once to analyze.</div>', unsafe_allow_html=True)

    with st.form("symptom_form", clear_on_submit=False):
        selected = st.multiselect("Symptoms", available_symptoms, placeholder="Search and select symptoms…", key="symptom_select")
        st.caption(f"{len(selected)} selected")
        submitted = st.form_submit_button("Analyze symptoms  →", type="primary", use_container_width=True)
    if submitted:
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analyzing symptoms…"):
                input_text = " ".join(x.replace(" ", "_") for x in selected)
                probs = symptom_model.predict_proba(vectorizer.transform([input_text]))[0]
                idx = int(probs.argmax()); disease = symptom_model.classes_[idx]; confidence = float(probs[idx])
                dvals = desc.loc[desc["Disease"] == disease, "Description"].values
                description = dvals[0] if len(dvals) else "No description available."
                row = prec[prec["Disease"] == disease]
                precautions = row.iloc[0, 1:].dropna().tolist() if not row.empty else []
                st.session_state.symptom_result = (disease, confidence, description, precautions)

    if st.session_state.symptom_result:
        disease, confidence, description, precautions = st.session_state.symptom_result
        st.markdown(f'<div class="card result"><div class="resultLabel">Model result</div><div class="resultValue">{html.escape(str(disease))}</div><div class="confidence">Confidence · {confidence:.1%}</div><div class="bar"><div style="width:{max(4,confidence*100):.1f}%"></div></div></div>', unsafe_allow_html=True)
        l,r = st.columns(2, gap="medium")
        with l:
            st.markdown(f'<div class="card panel"><div class="section">Description</div>{html.escape(str(description))}</div>', unsafe_allow_html=True)
        with r:
            items = "".join(f"<li>{html.escape(str(x))}</li>" for x in precautions) if precautions else "<li>No precaution listed in the dataset.</li>"
            st.markdown(f'<div class="card panel"><div class="section">Dataset precautions</div><ul style="color:#66768a;font-size:13px;line-height:1.7">{items}</ul></div>', unsafe_allow_html=True)
        if st.button("Clear result", key="clear_symptoms"):
            st.session_state.symptom_result = None
            st.rerun()
    st.markdown('<div class="notice"><b>Reminder:</b> This prediction is generated from the project dataset and should not be treated as a diagnosis.</div>', unsafe_allow_html=True)

# -----------------------------
# Image module helper
# -----------------------------
def render_image_module(kind):
    is_xray = kind == "xray"
    title = "Chest X-ray" if is_xray else "Skin Lesion"
    eyebrow = "Module 02 · Image classification" if is_xray else "Module 03 · Image classification"
    icon = "🫁" if is_xray else "🔬"
    state_key = "xray_result" if is_xray else "skin_result"
    uploader_key = "xray_file" if is_xray else "skin_file"
    model_name = "pneumonia model" if is_xray else "skin-lesion model"

    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Upload a JPG, JPEG or PNG image and submit once. Your previous result stays visible while you interact with the page.</div>', unsafe_allow_html=True)

    with st.form(f"{kind}_form", clear_on_submit=False):
        uploaded = st.file_uploader("Choose image", type=["jpg","jpeg","png"], key=uploader_key, label_visibility="visible")
        submit = st.form_submit_button(f"{icon}  Analyze {title} →", type="primary", use_container_width=True)

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left,right = st.columns([1,1.05], gap="large")
        with left:
            st.markdown('<div class="section">Preview</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown(f'<div class="successBox">✓ Image ready · {image.width} × {image.height}px</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section">Analysis</div>', unsafe_allow_html=True)
            if submit:
                try:
                    with st.spinner(f"Running {model_name}…"):
                        model = load_xray_model() if is_xray else load_skin_model()
                        index, confidence = predict_image(model, image)
                    if is_xray:
                        label = ["NORMAL", "PNEUMONIA"][index]
                    else:
                        codes = ["bkl","nv","df","mel","vasc","bcc","akiec"]
                        names = {"bkl":"Benign keratosis-like lesions","nv":"Melanocytic nevi","df":"Dermatofibroma","mel":"Melanoma","vasc":"Vascular lesions","bcc":"Basal cell carcinoma","akiec":"Actinic keratoses / intraepithelial carcinoma"}
                        label = names[codes[index]]
                    st.session_state[state_key] = (label, confidence, uploaded.name)
                except Exception as exc:
                    st.error(f"Could not run the model: {exc}")

            result = st.session_state.get(state_key)
            if result and result[2] == uploaded.name:
                label, confidence, _ = result
                accent = "#dc2626" if is_xray and label == "PNEUMONIA" else "#0f766e"
                st.markdown(f'<div class="card result"><div class="resultLabel">Prediction</div><div class="resultValue" style="color:{accent}">{label}</div><div class="confidence">Model confidence · {confidence:.1%}</div><div class="bar"><div style="width:{max(4,confidence*100):.1f}%;background:{accent}"></div></div></div>', unsafe_allow_html=True)
                if st.button("Clear result", key=f"clear_{kind}"):
                    st.session_state[state_key] = None
                    st.rerun()
            else:
                st.markdown('<div class="card panel" style="min-height:220px;text-align:center;padding-top:65px"><div style="font-size:40px">✨</div><b>Ready when you are</b><div class="smallMuted" style="margin-top:6px">Press the Analyze button after choosing an image.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="notice"><b>Medical safety:</b> This is an AI image-classification demonstration, not a medical diagnosis. A qualified healthcare professional should interpret medical images and clinical concerns.</div>', unsafe_allow_html=True)


if module == "Chest X-ray":
    render_image_module("xray")
elif module == "Skin Lesion":
    render_image_module("skin")
