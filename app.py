import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Page Header
st.set_page_config(page_title="Abstractive Text Summarizer", layout="centered")
st.title("Abstractive Text Summarizer")
st.caption("CCS335 Cloud Computing Mini Project | Transformer-based Summarization")

# Load Tokenizer and Model
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    return tokenizer, model

with st.spinner("Loading BART model into cloud memory..."):
    tokenizer, model = load_model()

# User Input
text_input = st.text_area("Input Text / News Article:", height=220, placeholder="Paste your article or long text here...")

# Generation Logic
if st.button("Generate Summary", type="primary"):
    if text_input.strip():
        with st.spinner("Generating summary..."):
            inputs = tokenizer(text_input, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=60,
                    min_length=10,
                    num_beams=4,
                    early_stopping=True
                )
            generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            st.success("Summary Generated Successfully!")
            st.subheader("Generated Summary:")
            st.write(generated_summary)
    else:
        st.warning("Please enter valid text to summarize.")