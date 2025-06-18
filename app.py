# pip install transformers Pillow torch into the shellpip install transformers Pillow

import os
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import torch
import streamlit as st

# === Step 1: Set up Hugging Face Token ===
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]  # Demo token for workshop

# === Step 2: Load Image (change filename as needed) ===
st.title("Food Image Captioning and QA")

uploaded_file = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    # === Step 3: Load BLIP model and generate caption ===
    with st.spinner("Generating caption..."):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)
        st.success("Caption generated!")
        st.write(f"📝 **Caption:** {caption}")

        # === Step 4: Use QA model to extract ingredients and steps ===
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        questions = [
            ("What are the ingredients?", "ingredients"),
            ("What are the cooking actions?", "actions"),
            ("Where does the dish originate?", "origin")
        ]
        st.subheader("🔍 Ask about the dish:")
        for q, key in questions:
            if st.button(q, key=key):
                with st.spinner(f"Answering: {q}"):
                    result = qa_pipeline(question=q, context=caption)
                    st.write(f"**Q:** {q}")
                    st.write(f"**A:** {result['answer']}")
else:
    st.info("Please upload an image to get started.")