import os
# ======================
# SETTING AWAL (WAJIB PALING ATAS)
# ======================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
from PIL import Image
import pickle
import json
import pandas as pd
import plotly.express as px
import h5py
import time
import gdown

# ======================
# IMPORT TENSORFLOW
# ======================
try:
    import tensorflow as tf
except ImportError:
    st.error("TensorFlow belum terinstall. Pastikan library 'tensorflow' ada di requirements.txt")
    st.stop()

# ======================
# FUNGSI DOWNLOAD DARI DRIVE
# ======================
def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            # Menggunakan gdown untuk download file dari Google Drive
            gdown.download(url, output_path, quiet=False)
        except Exception as e:
            st.error(f"Gagal mendownload file {output_path}: {e}")

# ======================
# PATH CONFIG & AUTO DOWNLOAD
# ======================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL_FINAL = os.path.join(BASE_PATH, "Model_Final")
PATH_MODEL_PRE = os.path.join(BASE_PATH, "Model_pre-trained")

# Mapping File ID dari Link Drive Anda
CONFIG_DRIVE = {
    "cnn_model": ("1WIAprknLskJn_b8LvrPLbnaMiYJNK4ke", os.path.join(PATH_MODEL_FINAL, "fish_classifier_final.h5")),
    "cnn_label": ("1Xcf8GNMfftT_R-B8D82BJb3qT9Oc45rd", os.path.join(PATH_MODEL_FINAL, "fish_labels_final.json")),
    "vgg_model": ("15vHRgheLLKEmmgO2V4i1OdJfn4YEM9Q0", os.path.join(PATH_MODEL_PRE, "VGG16_fast_best_FIXED.keras")),
    "mobilenet": ("19W-yCDUvyYeqaxPRwRupNrYXT0kDvkbT", os.path.join(PATH_MODEL_PRE, "MobileNetV2_fast_best_FIXED.keras")),
    "pkl_label": ("1zgBIbhUiOGciunNf3_EfsBY5SPb30CGs", os.path.join(PATH_MODEL_PRE, "class_names.pkl")),
}

# Jalankan proses download otomatis saat aplikasi start
for key, (fid, fpath) in CONFIG_DRIVE.items():
    download_from_drive(fid, fpath)

MODEL_PATHS = {
    "Custom CNN": CONFIG_DRIVE["cnn_model"][1],
    "VGG16 (Transfer Learning)": CONFIG_DRIVE["vgg_model"][1],
    "MobileNetV2 (Lightweight)": CONFIG_DRIVE["mobilenet"][1],
}

LABEL_PATHS = {
    "cnn": CONFIG_DRIVE["cnn_label"][1],
    "pretrained": CONFIG_DRIVE["pkl_label"][1],
}

# ======================
# PAGE CONFIG & CSS
# ======================
st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="wide")

st.markdown("""
<style>
.sidebar-header { display: flex; align-items: center; justify-content: center; gap: 14px; margin-top: 10px; margin-bottom: 6px; }
.sidebar-fish { font-size: 72px; line-height: 1; }
.sidebar-title { font-size: 28px; font-weight: 800; color: white; line-height: 1.1; }
.sidebar-subtitle { text-align: center; font-size: 13px; color: #8b949e; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ======================
# UTILITIES (PATCH & LOAD)
# ======================
def deep_clean_config(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") == "InputLayer":
            cfg = obj.get("config", {})
            if "shape" in cfg: cfg["batch_input_shape"] = cfg.pop("shape")
            obj["config"] = cfg
        if "dtype" in obj and isinstance(obj["dtype"], dict):
            obj["dtype"] = obj["dtype"].get("config", {}).get("name", "float32")
        obj.pop("registered_name", None)
        obj.pop("module", None)
        return {k: deep_clean_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clean_config(i) for i in obj]
    return obj

def safe_load_model(path):
    if not os.path.exists(path): return None
    try:
        # Percobaan load pertama
        return tf.keras.models.load_model(path, compile=False)
    except:
        try:
            # Percobaan load kedua dengan patching config (untuk model .h5 lama)
            with h5py.File(path, "r+") as f:
                if "model_config" in f.attrs:
                    raw = f.attrs["model_config"]
                    if isinstance(raw, bytes): raw = raw.decode("utf-8")
                    cfg = deep_clean_config(json.loads(raw))
                    f.attrs["model_config"] = json.dumps(cfg).encode("utf-8")
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            st.error(f"Gagal memuat {os.path.basename(path)}: {e}")
            return None

@st.cache_resource
def load_all_resources():
    progress = st.progress(0)
    status = st.empty()
    models = {}
    labels = {"cnn": [], "pretrained": []}

    # Load labels
    status.text("📂 Memuat file label...")
    if os.path.exists(LABEL_PATHS["cnn"]):
        with open(LABEL_PATHS["cnn"], "r") as f:
            labels["cnn"] = json.load(f)
    
    if os.path.exists(LABEL_PATHS["pretrained"]):
        with open(LABEL_PATHS["pretrained"], "rb") as f:
            labels["pretrained"] = pickle.load(f)

    progress.progress(25)

    # Load models
    for i, (name, path) in enumerate(MODEL_PATHS.items()):
        status.text(f"🧠 Memuat model: {name}")
        model = safe_load_model(path)
        if model: 
            models[name] = model
        progress.progress(25 + int((i + 1) / len(MODEL_PATHS) * 75))
    
    status.empty()
    progress.empty()
    return models, labels

with st.spinner("Harap tunggu, sedang memproses model dan label..."):
    loaded_models, all_labels = load_all_resources()

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-fish">🐟</div>
        <div class="sidebar-title">Fish Classifier</div>
    </div>
    <div class="sidebar-subtitle">Made By Ferdy Rizal Mahendra Putra<br>202210370311161<br>Machine Learning C</div>
    """, unsafe_allow_html=True)

    st.divider()
    if loaded_models:
        selected_models = st.multiselect("Pilih model untuk dibandingkan:", list(loaded_models.keys()), default=list(loaded_models.keys()))
    else:
        st.error("Model tidak ditemukan.")
        selected_models = []
    
    st.divider()
    st.write(f"**TensorFlow Version:** {tf.__version__}")

# ======================
# MAIN CONTENT
# ======================
st.title("🐟 Dashboard Klasifikasi Jenis Ikan")
st.info("**Model tersedia: CNN Custom | VGG16 | MobileNetV2**")
st.markdown("---")

if not selected_models:
    st.warning("Silakan pilih minimal satu model di sidebar sebelah kiri.")
    st.stop()

uploaded_file = st.file_uploader("Upload gambar ikan (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🖼️ Gambar Input")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("📊 Hasil Prediksi")
        tabs = st.tabs(selected_models)
        summary = []

        for i, model_name in enumerate(selected_models):
            with tabs[i]:
                model = loaded_models[model_name]
                current_labels = all_labels["cnn"] if model_name == "Custom CNN" else all_labels["pretrained"]

                # Cek input shape model
                try:
                    h, w = model.input_shape[1], model.input_shape[2]
                except:
                    h, w = (224, 224)

                # Preprocessing
                img_resized = img.resize((w, h))
                arr = tf.keras.preprocessing.image.img_to_array(img_resized)
                
                # Normalisasi (hanya untuk pretrained)
                if model_name != "Custom CNN":
                    arr = arr / 255.0
                
                arr = np.expand_dims(arr, axis=0)

                # Prediksi
                preds = model.predict(arr, verbose=0)[0]
                # Jika output belum softmax, terapkan softmax
                probs = tf.nn.softmax(preds).numpy() if preds.max() > 1 else preds

                idx = np.argmax(probs)
                label = current_labels[idx] if idx < len(current_labels) else f"Class {idx}"
                conf = probs[idx]

                # Tampilan Metrik
                c1, c2 = st.columns(2)
                c1.metric("Hasil Prediksi", label)
                c2.metric("Tingkat Keyakinan", f"{conf*100:.2f}%")

                # Grafik Probabilitas
                df_chart = pd.DataFrame({
                    "Label": current_labels, 
                    "Probabilitas": probs
                }).sort_values("Probabilitas").tail(5)
                
                fig = px.bar(df_chart, x="Probabilitas", y="Label", orientation="h", 
                             color="Probabilitas", color_continuous_scale="Blues")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)

                summary.append({
                    "Model": model_name, 
                    "Prediksi": label, 
                    "Confidence": f"{conf*100:.2f}%"
                })

        st.divider()
        st.subheader("📝 Ringkasan Hasil Semua Model")
        st.table(pd.DataFrame(summary))