import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img_process
from PIL import Image
import pathlib
import os

@st.cache_resource
def load_keras_model():
    try:
        base_dir = pathlib.Path(__file__).parent.resolve()
        model_path = base_dir / "model_xception.keras"

        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        print(f"Error detail: {str(e)}")
        return None

st.title("🏥 Deteksi Pneumonia Menggunakan Xception")
st.write("Upload gambar rontgen dada untuk mendeteksi apakah pasien terkena pneumonia atau tidak.")

# Load model
model = load_keras_model()
if model is None:
    st.stop()

class_names = {0: "NORMAL", 1: "PNEUMONIA"}

# Upload gambar
uploaded_file = st.file_uploader("📤 Pilih gambar rontgen (JPG/PNG/JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='🖼️ Gambar yang diupload', width=400)
    
    img = img.resize((150, 150))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = img_process.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    with st.spinner("🧠 Model sedang memproses..."):
        try:
            prediction = model.predict(img_array, verbose=0)
            prediction_value = float(prediction[0][0])  # ambil nilai sigmoid

            class_idx = 1 if prediction_value >= 0.5 else 0
            confidence = prediction_value if class_idx == 1 else 1 - prediction_value
            label = class_names[class_idx]

            # Hasil prediksi
            emoji = "🔴" if class_idx == 1 else "🟢"
            st.success(f"{emoji} Hasil Prediksi: **{label}** (Confidence: {confidence:.2f})")

            if class_idx == 1:
                st.warning("⚠️ Terdeteksi tanda-tanda pneumonia pada gambar rontgen. Konsultasikan dengan dokter untuk diagnosis lebih lanjut.")
            else:
                st.info("ℹ️ Tidak terdeteksi tanda-tanda pneumonia pada gambar rontgen ini.")

        except Exception as e:
            st.error(f"❌ Terjadi error saat prediksi: {str(e)}")
            import traceback
            print(traceback.format_exc())
