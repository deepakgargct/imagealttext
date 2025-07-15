import streamlit as st
import pandas as pd
import requests
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import time

st.set_page_config(page_title="ALT Text Generator", layout="wide")
st.title("🧠 ALT Text Generator using Ollama + LLaVA")
st.markdown("Upload a CSV with image URLs. The app generates ALT text using your local Ollama model.")

# --- Configuration options ---
MAX_RETRIES = 3
TIMEOUT = 10
MAX_THREADS = 6

check_existing_alts = st.checkbox("🕵️‍♀️ Skip images with existing ALT text from HTML page", value=True)

# --- Step 1: HTML scraping to detect existing ALT text ---
def check_alt_tag(image_url):
    try:
        page_url = image_url.rsplit("/", 1)[0]  # Guess parent page
        response = requests.get(page_url, timeout=TIMEOUT)
        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all("img", src=True)
        for tag in img_tags:
            if image_url.endswith(tag["src"].split("/")[-1]) and tag.get("alt"):
                return True  # Image has alt text
        return False
    except Exception:
        return False

# --- Step 2: Download image and convert to base64 ---
def convert_to_base64(image_url):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(image_url, timeout=TIMEOUT)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
        except Exception:
            time.sleep(1)
    return None

# --- Step 3: Send to Ollama ---
def generate_alt_text(base64_image):
    for attempt in range(MAX_RETRIES):
        try:
            payload = {
                "model": "llava",
                "prompt": "Please provide a functional, objective description of the provided image in no more than around 30 words so that someone who could not see it would be able to imagine it. Use an object-action-context style. Transcribe any visible text.",
                "stream": False,
                "images": [base64_image]
            }
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception:
            time.sleep(1)
    return "[Error: Failed to generate ALT text after retries]"

# --- Step 4: Process image URL ---
def process_image(img_url):
    if check_existing_alts and check_alt_tag(img_url):
        return {"image_url": img_url, "alt_text": "[Skipped: Existing ALT text found]"}
    
    base64_img = convert_to_base64(img_url)
    if base64_img:
        alt_text = generate_alt_text(base64_img)
    else:
        alt_text = "[Error: Could not fetch image]"
    
    return {"image_url": img_url, "alt_text": alt_text}

# --- Step 5: Upload CSV ---
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

        # --- Image Previews ---
        st.subheader("🖼️ ALT Text Results")
        cols = st.columns(3)
        for i, row in result_df.iterrows():
            col = cols[i % 3]
            with col:
                try:
                    st.image(row["image_url"], width=200)
                except Exception:
                    st.warning("Could not load image preview")
                st.markdown(f"**ALT Text:** {row['alt_text']}")

        # --- CSV Download ---
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results CSV", data=csv_data, file_name="alt_text_results.csv", mime="text/csv")
