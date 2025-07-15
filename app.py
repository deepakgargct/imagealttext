import streamlit as st
import pandas as pd
import requests
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# ------------------ Config ------------------
st.set_page_config(page_title="ALT Text Generator", layout="wide")
st.title("🧠 ALT Text Generator using Ollama + LLaVA")
st.markdown("Upload a CSV with image URLs. The app will use your local Ollama (`llava`) model to generate descriptive ALT text for each image.")

MAX_RETRIES = 3
TIMEOUT = 10
MAX_THREADS = 6

# ------------------ Ollama Status Checker ------------------
with st.expander("🧪 Check Ollama Server & Models", expanded=False):
    if st.button("🔍 Check Ollama Status"):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m['name'] for m in r.json().get('models', [])]
                if models:
                    st.success("✅ Ollama is running. Available models:\n\n" + "\n".join(models))
                else:
                    st.warning("⚠️ Ollama is running, but no models are installed.")
            else:
                st.error(f"⚠️ Unexpected response from Ollama: {r.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Could not connect to Ollama at http://localhost:11434\n\nError: {str(e)}")

# ------------------ Options ------------------
check_existing_alts = st.checkbox("🕵️‍♀️ Skip images with existing ALT text (HTML scraping)", value=True)

# ------------------ Helpers ------------------

def check_alt_tag(image_url):
    try:
        page_url = image_url.rsplit("/", 1)[0]
        response = requests.get(page_url, timeout=TIMEOUT)
        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all("img", src=True)
        for tag in img_tags:
            if image_url.endswith(tag["src"].split("/")[-1]) and tag.get("alt"):
                return True
        return False
    except Exception:
        return False

def convert_to_base64(image_url):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(image_url, timeout=TIMEOUT)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
        except Exception:
            time.sleep(1)
    return None

def generate_alt_text(base64_image):
    for attempt in range(MAX_RETRIES):
        try:
            payload = {
                "model": "llava:latest",
                "prompt": "Please provide a functional, objective description of the provided image in no more than around 30 words so that someone who could not see it would be able to imagine it. Use an object-action-context style. Transcribe any visible text.",
                "stream": False,
                "images": [base64_image]
            }
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception:
            time.sleep(1)
    return "[Error: Failed to generate ALT text after retries]"

def process_image(img_url):
    if check_existing_alts and check_alt_tag(img_url):
        return {"image_url": img_url, "alt_text": "[Skipped: Existing ALT text found]"}
    
    base64_img = convert_to_base64(img_url)
    if base64_img:
        alt_text = generate_alt_text(base64_img)
    else:
        alt_text = "[Error: Could not fetch image]"
    
    return {"image_url": img_url, "alt_text": alt_text}

# ------------------ Upload CSV ------------------
uploaded_file = st.file_uploader("📤 Upload CSV with Image URLs", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if "image_url" not in df.columns:
        st.error("CSV must contain a column named 'image_url'")
    else:
        urls = df["image_url"].dropna().unique().tolist()
        st.info(f"Processing {len(urls)} images using up to {MAX_THREADS} threads...")

        results = []
        progress = st.progress(0, text="🔄 Starting...")

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future_to_url = {executor.submit(process_image, url): url for url in urls}
            for i, future in enumerate(as_completed(future_to_url)):
                result = future.result()
                results.append(result)
                progress.progress((i + 1) / len(urls), text=f"Processed {i + 1}/{len(urls)}")

        progress.empty()

        result_df = pd.DataFrame(results)

        # ------------------ Image Previews ------------------
        st.subheader("🖼️ ALT Text Results")
        cols = st.columns(3)

        for i, row in result_df.iterrows():
            col = cols[i % 3]
            with col:
                try:
                    st.image(row["image_url"], width=200, caption="Preview")
                except Exception:
                    st.warning("⚠️ Could not load image preview")
                st.markdown(f"**ALT Text:** {row['alt_text']}")

        # ------------------ Download CSV ------------------
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results CSV", data=csv_data, file_name="alt_text_results.csv", mime="text/csv")
