import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load Grounding DINO
@st.cache_resource
def load_grounding_dino():
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

# Load OWL-ViT
@st.cache_resource
def load_owlvit():
    model_id = "google/owlvit-base-patch32"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

# Fungsi prediksi
def predict(image, text_prompts, processor, model, device, model_type="dino", threshold=0.3):
    if not text_prompts or not isinstance(text_prompts, list):
        raise ValueError("`text_prompts` harus berupa list of strings.")

    if model_type == "dino":
        text_input = ", ".join(text_prompts)
    elif model_type == "owlvit":
        text_input = text_prompts
    else:
        raise ValueError("model_type harus 'dino' atau 'owlvit'.")

    inputs = processor(images=image, text=text_input, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = [image.size[::-1]]

    if model_type == "dino":
        results = processor.post_process_grounded_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    else:
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)[0]

    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()

    if isinstance(results["labels"], torch.Tensor):
        label_indices = results["labels"].cpu().tolist()
    else:
        label_indices = results["labels"]

    if model_type == "dino":
        labels = [text_prompts[0]] * len(boxes)  # semua pakai prompt tunggal
    else:
        labels = [text_prompts[i] if i < len(text_prompts) else "unknown" for i in label_indices]

    return boxes, labels, scores

# Gambar bounding box
def draw_boxes_on_image(image, boxes, labels, scores):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0), f"{label} {score:.2f}", fill="red", font=font)

    return image

# Antarmuka Streamlit
def main():
    st.title("🔍 Deteksi Objek dengan Grounding DINO & OWL-ViT")

    st.sidebar.header("⚙️ Pengaturan")
    raw_prompts = st.sidebar.text_area("Masukkan prompt teks (pisahkan koma)", "person, bicycle, car")
    text_prompts = [t.strip() for t in raw_prompts.split(",") if t.strip()]
    threshold = st.sidebar.slider("Threshold deteksi", 0.1, 1.0, 0.3, 0.05)

    uploaded_file = st.sidebar.file_uploader("📁 Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file and text_prompts:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Gambar Input", use_container_width=True)

        # Load models
        dino_processor, dino_model, dino_device = load_grounding_dino()
        owl_processor, owl_model, owl_device = load_owlvit()

        # Grounding DINO
        st.subheader("📦 Hasil Deteksi - Grounding DINO")
        boxes, labels, scores = predict(image, text_prompts, dino_processor, dino_model, dino_device, model_type="dino", threshold=threshold)
        dino_img = draw_boxes_on_image(image.copy(), boxes, labels, scores)
        st.image(dino_img, caption="Grounding DINO", use_container_width=True)
        for label, score, box in zip(labels, scores, boxes):
            st.write(f"{label}: {score:.2f} - Box: {box}")

        # OWL-ViT
        st.subheader("📦 Hasil Deteksi - OWL-ViT")
        boxes, labels, scores = predict(image, text_prompts, owl_processor, owl_model, owl_device, model_type="owlvit", threshold=threshold)
        owl_img = draw_boxes_on_image(image.copy(), boxes, labels, scores)
        st.image(owl_img, caption="OWL-ViT", use_container_width=True)
        for label, score, box in zip(labels, scores, boxes):
            st.write(f"{label}: {score:.2f} - Box: {box}")

if __name__ == "__main__":
    main()
