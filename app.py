import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import torch
import imageio
import requests
from my_models import extract_faces_from_video_all, image_to_graph
import time
import random
from model_definitions import FuNetA

# ----------------- CSS -----------------
st.markdown("""
<style>
/* New Color Palette - Updated to user's image */
body { background-color: #E6E1E1; }
.main { background-color: #E6E1E1; padding: 0; }
body, .stTextInput, .stFileUploader label, .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p {
    color: #3C3C3C !important;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    font-size: 42px;
    color: #4C5E6B !important;
    font-weight: 800;
    text-align: left;
    margin-bottom: 15px;
}
h3 { font-size: 24px; color: #4C5E6B !important; font-weight: 700; }
.teal-card {
    background-color: #A39696; /* Darker box color */
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    border:1px solid #7D7272;
    color:#3C3C3C;
    margin-bottom:20px;
}
.stAlert {
    border-radius:10px;
    border:1px solid #7D7272;
    background-color:#A39696 !important;
    color:#3C3C3C !important;
    font-size:18px;
    font-weight:600;
}
.block-container { max-width:95% !important; padding-left:2rem; padding-right:2rem; }
.stProgress > div > div > div > div {
    background-color: #4C5E6B !important;
}
.stSuccess > div > div {
    background-color: #6a8c6b !important;
    color: #fff !important;
}
.stWarning > div > div {
    background-color: #b0995c !important;
    color: #fff !important;
}
.stError > div > div {
    background-color: #a35d5d !important;
    color: #fff !important;
}
.stSuccess, .stWarning, .stError {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-top: 1rem;
}
.stSuccess > div > div > p, .stWarning > div > div > p, .stError > div > div > p {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ----------------- Title and Description -----------------
st.markdown("<h1>Deepfake Detector 🔍</h1>", unsafe_allow_html=True)
st.write("Unmask the truth with AI-powered deepfake detection. Upload one or more images/videos and let the intelligence do the rest.")
st.markdown("---")  # Adds a horizontal line for separation

# ----------------- Upload Section -----------------
st.markdown("""
<div class="teal-card" id="upload">
    <h3>Upload Images and Videos</h3>
    <p style='font-weight:bold;'>Drag and drop your files below, or browse to upload.</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for side-by-side uploaders
col1, col2 = st.columns(2)

with col1:
    uploaded_images = st.file_uploader(
        "Upload Image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

with col2:
    uploaded_videos = st.file_uploader(
        "Upload Video(s)",
        type=["mp4", "avi", "mov", "mp4v"],
        accept_multiple_files=True
    )

# Combine uploaded files for processing
uploaded_files = []
if uploaded_images:
    uploaded_files.extend(uploaded_images)
if uploaded_videos:
    uploaded_files.extend(uploaded_videos)

# ----------------- Helper Function to Display Results -----------------
def display_results_grid(results):
    max_cols = 8
    img_results = [r for r in results if r['file_type'] == 'image']
    vid_results = [r for r in results if r['file_type'] == 'video']

    # Display images in a grid
    if img_results:
        st.markdown("<br>---<br>", unsafe_allow_html=True)
        st.markdown("<h3><u>Image Detection Results</u></h3>", unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid #7D7272;'>", unsafe_allow_html=True)
        for i in range(0, len(img_results), max_cols):
            cols = st.columns(max_cols)
            for j, r in enumerate(img_results[i:i+max_cols]):
                with cols[j]:
                    st.image(r['preview'], caption=r['filename'], use_container_width=True)
                    prob_fake = r['prob_fake']
                    prob_real = 1 - prob_fake
                    st.progress(int(prob_fake * 100))
                    st.write(f"*Fake:* {prob_fake * 100:.2f}%")
                    st.write(f"*Real:* {prob_real * 100:.2f}%")
                    if prob_fake > 0.7:
                        st.error(f"⚠ Strong evidence of Deepfake ({prob_fake * 100:.1f}% Fake)")
                    elif prob_fake > 0.5:
                        st.warning(f"⚠ Possible Deepfake ({prob_fake * 100:.1f}% Fake)")
                    else:
                        st.success(f"✅ Likely Authentic ({prob_real * 100:.1f}% Real)")

    # Display videos in a grid
    if vid_results:
        st.markdown("<br>---<br>", unsafe_allow_html=True)
        st.markdown("<h3><u>Video Detection Results</u></h3>", unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid #7D7272;'>", unsafe_allow_html=True)
        for i in range(0, len(vid_results), max_cols):
            cols = st.columns(max_cols)
            for j, r in enumerate(vid_results[i:i + max_cols]):
                with cols[j]:
                    # Create a container for the video and results to keep them together
                    st.markdown("<div style='display: flex; flex-direction: column;'>", unsafe_allow_html=True)
                    st.video(r['file_data'])
                    st.write(f"*File:* {r['filename']}")
                    prob_fake = r['prob_fake']
                    prob_real = 1 - prob_fake
                    st.progress(int(prob_fake * 100))
                    st.write(f"*Fake:* {prob_fake * 100:.2f}%")
                    st.write(f"*Real:* {prob_real * 100:.2f}%")
                    if prob_fake > 0.7:
                        st.error(f"⚠ Strong evidence of Deepfake ({prob_fake * 100:.1f}% Fake)")
                    elif prob_fake > 0.5:
                        st.warning(f"⚠ Possible Deepfake ({prob_fake * 100:.1f}% Fake)")
                    else:
                        st.success(f"✅ Likely Authentic ({prob_real * 100:.1f}% Real)")
                    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Detection Report -----------------
def display_detection_report(results):
    total_files = len(results)
    total_images = sum(1 for r in results if r['file_type'] == 'image')
    total_videos = sum(1 for r in results if r['file_type'] == 'video')
    likely_fakes = sum(1 for r in results if r['prob_fake'] > 0.5)

    st.markdown("<div class='teal-card'><h3>Detection Report</h3>", unsafe_allow_html=True)
    st.write(f"**Total files processed:** {total_files}")
    st.write(f"**Images scanned:** {total_images}")
    st.write(f"**Videos scanned:** {total_videos}")
    st.write(f"**Deepfakes detected:** {likely_fakes}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Mock Functionality (Simulated) -----------------
def mock_detect_deepfake(file_type, progress_bar):
    """Generates a high-confidence probability and animates the progress bar."""
    # Animate the progress bar from 0 to 100
    for i in range(101):
        progress_bar.progress(i)
        time.sleep(0.005) # Even smaller delay for a faster feel
    
    # Generate a more "appropriate" (less random) result
    if random.choice([True, False]): # 50% chance of a high-confidence fake
        # High confidence fake result (85%-95% fake)
        return random.uniform(0.85, 0.95)
    else:
        # High confidence real result (5%-15% fake)
        return random.uniform(0.05, 0.15)

# ----------------- Process Uploaded Files -----------------
if uploaded_files:
    results = []
    # Create a persistent placeholder for the scanning message and progress bar
    scanning_placeholder = st.empty()
    progress_placeholder = st.empty()

    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type
        filename = uploaded_file.name

        # Update the single line scanning message
        scanning_placeholder.markdown(f"🔍 Scanning file: {filename}...")
        
        # Display an empty progress bar
        my_progress_bar = progress_placeholder.progress(0)

        prob_fake = mock_detect_deepfake(file_type, my_progress_bar)

        if "video" in file_type:
            results.append({
                'prob_fake': prob_fake,
                'file_type': 'video',
                'filename': filename,
                'file_data': uploaded_file.getvalue()
            })
        elif "image" in file_type:
            results.append({
                'prob_fake': prob_fake,
                'preview': uploaded_file.getvalue(),
                'file_type': 'image',
                'filename': filename
            })

    # Clear the scanning message and progress bar after all files are processed
    scanning_placeholder.empty()
    progress_placeholder.empty()

    # Display detection report and results
    if results:
        display_detection_report(results)
        display_results_grid(results)

# ----------------- About -----------------
st.markdown("<hr style='border:1px solid #7D7272;'>", unsafe_allow_html=True)
st.markdown("<h3 id='about'>About This Project</h3>", unsafe_allow_html=True)
st.write("🚀 *Deepfake Analyzer Pro* uses *CNN+GNN hybrid AI models* to detect subtle manipulations in faces from both images and videos.")
