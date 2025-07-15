import streamlit as st
import pandas as pd
import requests
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="ALT Text Generator with LLaVA", layout="wide")
st.title("🧠 ALT Text Generator using Ollama + LLaVA (Parallel)")
st.markdown("Upload a CSV with image URLs. This app will use your local Ollama (`llava`) model to generate ALT text using parallel processing.")

# Convert image URL to base64
def convert_to_base64(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        base64_image = base64.b64encode(response.content).decode("utf-8")
        return base64_image
    except Exception:
        return None

# Send base64 image to Ollama LLaVA
def generate_alt_text(base64_image):
    try:
        payload = {
            "model": "llava",
            "prompt": "Please provide a functional, objective description of the provided image in no more than around 30 words so that someone who could not see it would be able to imagine it. If possible, follow an “object-action-context” framework.",
            "stream": False,
            "images": [base64_image]
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"[Error: {str(e)}]"

# Combined task per image
def process_image(img_url):
    base64_img = convert_to_base64(img_url)
    if base64_img:
        alt_text = generate_alt_text(base64_img)
    else:
        alt_text = "[Error: Could not fetch image]"
    return {"image_url": img_url, "alt_text": alt_text}

# UI: Upload CSV
uploaded_file = st.file_uploader("📤 Upload CSV with Image URLs", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "image_url" not in df.columns:
        st.error("CSV must have a column named 'image_url'")
    else:
        urls = df["image_url"].dropna().unique().tolist()
        st.info(f"Found {len(urls)} image URLs. Starting parallel processing...")
        results = []

        progress_text = st.empty()
        progress_bar = st.progress(0)

        max_threads = min(6, len(urls))  # Use up to 6 threads
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_to_url = {executor.submit(process_image, url): url for url in urls}

            for i, future in enumerate(as_completed(future_to_url)):
                result = future.result()
                results.append(result)
                progress_bar.progress((i + 1) / len(urls))
                progress_text.text(f"Processed {i + 1} / {len(urls)}")

        progress_bar.empty()
        progress_text.empty()

        result_df = pd.DataFrame(results)

        # UI: Display Results with Previews
        st.subheader("🖼️ ALT Text Results")
        cols = st.columns(3)

        for i, row in result_df.iterrows():
            col = cols[i % 3]
            with col:
                st.image(row["image_url"], width=200, caption="Preview")
                st.markdown(f"**ALT Text:** {row['alt_text']}", unsafe_allow_html=True)

        # UI: Download CSV
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results CSV", data=csv_data, file_name="alt_text_results.csv", mime="text/csv")
