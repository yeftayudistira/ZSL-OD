import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

@st.cache_resource
def load_grounding_dino():
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

@st.cache_resource
def load_owlvit():
    model_id = "google/owlvit-base-patch32"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

def predict(image, text_prompts, processor, model, device, threshold=0.3):
    inputs = processor(images=image, text=text_prompts, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=[image.size[::-1]]
    )[0]

    boxes = results["boxes"].cpu().tolist()
    labels = [text_prompts[i] for i in results["labels"].cpu().tolist()]
    scores = results["scores"].cpu().tolist()

    return boxes, labels, scores

def main():
    st.title("Deteksi Objek dengan GroundingDINO & OWL-ViT")

    st.sidebar.header("Pengaturan")
    text_prompts = st.sidebar.text_area("Masukkan prompt teks (pisahkan koma)", 
                                        "a person, a bicycle, a car").split(",")
    text_prompts = [t.strip() for t in text_prompts if t.strip()]

    st.sidebar.write("Upload gambar untuk deteksi objek")
    uploaded_file = st.sidebar.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file and text_prompts:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Gambar input", use_column_width=True)

        # Load models
        dino_processor, dino_model, dino_device = load_grounding_dino()
        owl_processor, owl_model, owl_device = load_owlvit()

        st.subheader("Hasil Deteksi GroundingDINO")
        boxes, labels, scores = predict(image, text_prompts, dino_processor, dino_model, dino_device)

        for box, label, score in zip(boxes, labels, scores):
            st.write(f"{label}: {score:.2f} - box: {box}")

        st.subheader("Hasil Deteksi OWL-ViT")
        boxes, labels, scores = predict(image, text_prompts, owl_processor, owl_model, owl_device)

        for box, label, score in zip(boxes, labels, scores):
            st.write(f"{label}: {score:.2f} - box: {box}")

if __name__ == "__main__":
    main()
