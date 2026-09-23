import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from pypdf import PdfReader
from collections import Counter

# ---------------------------------------------------------
# ROUGE Evaluation Functions (From project algorithm)
# ---------------------------------------------------------
def get_tokens(text):
    return text.lower().split()

def rouge_n(reference, generated, n):
    ref_tokens = get_tokens(reference)
    gen_tokens = get_tokens(generated)
    ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
    gen_ngrams = [tuple(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)]
    if not ref_ngrams or not gen_ngrams:
        return 0.0
    ref_count = Counter(ref_ngrams)
    gen_count = Counter(gen_ngrams)
    overlap = sum(min(ref_count[g], gen_count[g]) for g in gen_count)
    precision = overlap / len(gen_ngrams)
    recall = overlap / len(ref_ngrams)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def lcs_length(x, y):
    dp = [[0]*(len(y)+1) for _ in range(len(x)+1)]
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def rouge_l(reference, generated):
    ref_tokens = get_tokens(reference)
    gen_tokens = get_tokens(generated)
    if not ref_tokens or not gen_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, gen_tokens)
    precision = lcs / len(gen_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Abstractive Text Summarizer", layout="centered")
st.title("Abstractive Text Summarizer")
st.caption("CCS335 Cloud Computing Mini Project | Transformer Framework & ROUGE Evaluation")

# Load Tokenizer & Model
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    return tokenizer, model

with st.spinner("Loading BART transformer model..."):
    tokenizer, model = load_model()

# ---------------------------------------------------------
# Step 1: Input & File Upload
# ---------------------------------------------------------
st.subheader("1. Input Article or Document")
uploaded_file = st.file_uploader("Upload a file (.pdf or .txt):", type=["pdf", "txt"])

extracted_text = ""
if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        extracted_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        extracted_text = uploaded_file.read().decode("utf-8")

text_input = st.text_area(
    "Or paste text directly below:",
    value=extracted_text,
    height=200,
    placeholder="Paste article text here or upload a file above..."
)

# ---------------------------------------------------------
# Step 2: Customization & Evaluation Options
# ---------------------------------------------------------
st.subheader("2. Generation Controls & Metrics")
col1, col2 = st.columns(2)
with col1:
    target_length = st.slider("Max Summary Length (words):", min_value=20, max_value=150, value=60)
with col2:
    num_beams = st.slider("Beam Search Depth:", min_value=1, max_value=8, value=4)

reference_summary = st.text_area(
    "Reference Summary (Optional - used for live ROUGE score calculation):",
    height=80,
    placeholder="Paste expected reference summary here..."
)

# ---------------------------------------------------------
# Step 3: Summarization Logic
# ---------------------------------------------------------
if st.button("Generate Summary", type="primary"):
    if text_input.strip():
        with st.spinner("Summarizing text with BART model..."):
            inputs = tokenizer(text_input, return_tensors="pt", max_length=512, truncation=True)
            min_len = max(10, target_length // 3)
            
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=target_length,
                    min_length=min_len,
                    num_beams=num_beams,
                    early_stopping=True
                )
            generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            st.success("Summary Generated Successfully!")
            st.subheader("Generated Summary:")
            st.write(generated_summary)

            # Download Option
            st.download_button(
                label="📥 Download Summary (.txt)",
                data=generated_summary,
                file_name="generated_summary.txt",
                mime="text/plain"
            )

            # Calculate and display ROUGE scores if reference text is present
            if reference_summary.strip():
                r1 = rouge_n(reference_summary, generated_summary, 1)
                r2 = rouge_n(reference_summary, generated_summary, 2)
                rL = rouge_l(reference_summary, generated_summary)

                st.markdown("---")
                st.subheader("📊 ROUGE Score Evaluation")
                m1, m2, m3 = st.columns(3)
                m1.metric("ROUGE-1", f"{r1:.4f}")
                m2.metric("ROUGE-2", f"{r2:.4f}")
                m3.metric("ROUGE-L", f"{rL:.4f}")
    else:
        st.warning("Please enter text or upload a document to summarize.")
