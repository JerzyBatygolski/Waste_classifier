"""
app.py
======
Streamlit frontend for the waste classifier API.

How Streamlit works:
  Every user interaction (file upload, button click) re-runs this entire
  script from top to bottom. st.session_state is used to persist data
  between re-runs (e.g. keep the prediction result after upload).

Run locally:
  pip install streamlit requests pillow
  streamlit run app.py

  Then open: http://localhost:8501
"""

import io
import os

import requests
import streamlit as st
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.getenv("API_URL", "https://waste-classifier-api-spsy2vhjeq-lm.a.run.app")

# Preprocessing geometry - must match the API / training pipeline.
# Letterbox = resize keeping aspect ratio, pad the shorter side with grey.
TARGET_SIZE = 224
PAD_COLOR = (114, 114, 114)

# Color palette for the bar chart - one color per rank position.
# Winner (rank 0) gets a vivid color, the rest get progressively muted tones.
BAR_COLORS = [
    "#2E86AB",  # rank 1  - vivid blue   (winner)
    "#5BA4C8",  # rank 2
    "#88BFD9",  # rank 3
    "#A8C8D8",  # rank 4
    "#B8D0DC",  # rank 5
    "#C5D8E0",  # rank 6
    "#D0DDE3",  # rank 7
    "#D8E3E8",  # rank 8
    "#DEE7EB",  # rank 9
    "#E3EAED",  # rank 10 - very light   (last place)
]

# Human-readable class labels (shown in the chart instead of raw folder names)
CLASS_LABELS = {
    "battery":    "Battery",
    "biological": "Biological",
    "cardboard":  "Cardboard",
    "clothes":    "Clothes",
    "glass":      "Glass",
    "metal":      "Metal",
    "paper":      "Paper",
    "plastic":    "Plastic",
    "shoes":      "Shoes",
    "trash":      "Trash",
}


# ---------------------------------------------------------------------------
# Page config - must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Waste Classifier",
    page_icon=":recycle:",
    layout="centered",       # content centered, max ~700px wide
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Custom CSS
# Winner bar gets a thicker left accent border and bold text.
# All other bars use normal weight.
# Responsive: works on mobile because layout="centered" is already narrow.
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    [data-testid="stToolbar"] {
            visibility: hidden;
    }
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .tip {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .winner-box {
        background: #E6F1FB;
        border-left: 5px solid #2E86AB;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.4rem;
    }
    .winner-label {
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 0.2rem;
    }
    .winner-class {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0C447C;
    }
    .winner-conf {
        font-size: 1rem;
        color: #185FA5;
        margin-top: 0.1rem;
    }
    .bar-row {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
        gap: 8px;
    }
    .bar-label {
        width: 100px;
        font-size: 0.88rem;
        text-align: right;
        flex-shrink: 0;
        color: #333;
    }
    .bar-label-winner {
        font-weight: 700;
        color: #0C447C;
    }
    .bar-track {
        flex: 1;
        background: #f0f0f0;
        border-radius: 4px;
        height: 22px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }
    .bar-fill-winner {
        border-left: 4px solid #0C447C;
    }
    .bar-pct {
        width: 48px;
        font-size: 0.85rem;
        text-align: right;
        flex-shrink: 0;
        color: #555;
    }
    .bar-pct-winner {
        font-weight: 700;
        color: #0C447C;
    }
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# Streamlit re-runs the whole script on every interaction.
# session_state persists values across re-runs within the same browser session.
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None       # last API response dict
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None  # last uploaded image bytes (raw)
if "pending" not in st.session_state:
    st.session_state.pending = False     # True when a new image awaits classification
if "uploader_key" not in st.session_state:
    # Incrementing this key forces Streamlit to re-create the file_uploader
    # widget as a fresh, empty one - this is how we "clear" the uploader
    # after a successful classification.
    st.session_state.uploader_key = 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def letterbox_image(img, target_size=TARGET_SIZE, pad_color=PAD_COLOR):
    """
    Resize `img` so its longer side equals target_size while preserving the
    aspect ratio, then center it on a square target_size x target_size canvas
    filled with `pad_color` (grey). This mirrors the API / training
    preprocessing so the bytes we upload are already in the exact geometry the
    model expects (no aspect-ratio distortion).
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    # BILINEAR matches the default interpolation used by tf.image.resize /
    # Keras pipelines. Keep this consistent with how the model was trained.
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), pad_color)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def call_api(image_bytes):
    """
    Preprocess `image_bytes` (EXIF orientation fix + letterbox to
    TARGET_SIZE x TARGET_SIZE with grey padding), then POST the result to the
    /predict endpoint. Returns the parsed JSON dict on success, or raises.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)   # honor phone EXIF orientation tag
    img = letterbox_image(img)           # 224x224, grey (114,114,114) padding

    buf = io.BytesIO()
    img.save(buf, format="PNG")          # lossless: avoids JPEG edge artifacts
    payload = buf.getvalue()

    response = requests.post(
        API_URL + "/predict",
        files={"file": ("image.png", payload, "image/png")},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def render_bar_chart(all_scores):
    """
    Render a custom horizontal bar chart using HTML.
    Bars are sorted by score descending.
    Winner bar is visually emphasized (bold label, thicker border, vivid color).
    """
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    bars_html = ""
    for rank, (cls, score) in enumerate(sorted_scores):
        label      = CLASS_LABELS.get(cls, cls.capitalize())
        pct        = score * 100
        color      = BAR_COLORS[rank] if rank < len(BAR_COLORS) else BAR_COLORS[-1]
        bar_width  = max(pct, 0.5)   # min width so 0% bars are still visible
        is_winner  = rank == 0

        label_cls  = "bar-label bar-label-winner" if is_winner else "bar-label"
        fill_cls   = "bar-fill bar-fill-winner"   if is_winner else "bar-fill"
        pct_cls    = "bar-pct bar-pct-winner"     if is_winner else "bar-pct"
        height     = "28px" if is_winner else "22px"

        bars_html += f"""
        <div class="bar-row">
            <div class="{label_cls}">{label}</div>
            <div class="bar-track" style="height:{height};">
                <div class="{fill_cls}"
                     style="width:{bar_width:.1f}%;background:{color};height:100%;">
                </div>
            </div>
            <div class="{pct_cls}">{pct:.1f}%</div>
        </div>
        """

    st.markdown(bars_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI - title and instructions
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">Welcome to the Waste Classifier!</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a photo of waste and I will classify it for you.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="tip">Tip: for the best results, upload a photo of a single object '
    'on a plain, uniform background.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# File uploader
# Streamlit re-runs the script when a new file is uploaded.
# uploaded_file is None if nothing was uploaded yet.
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    label="Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",   # hide the label, the subtitle serves as instruction
    key="uploader_" + str(st.session_state.uploader_key),
)

# ---------------------------------------------------------------------------
# Handle a new upload: store the raw bytes in session_state and flag for
# classification. We deliberately do NOT classify here - we want the image to
# render first (display block below) so it appears immediately, before the
# potentially slow API call.
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    if image_bytes != st.session_state.image_bytes:
        st.session_state.image_bytes = image_bytes
        st.session_state.result = None
        st.session_state.pending = True

# ---------------------------------------------------------------------------
# Display the uploaded image immediately from session_state.
# Gated on image_bytes (not on result), so it shows the moment the file is
# uploaded and persists across every rerun - including after the uploader
# widget is cleared.
# ---------------------------------------------------------------------------

if st.session_state.image_bytes is not None:
    try:
        img = Image.open(io.BytesIO(st.session_state.image_bytes))
        img = ImageOps.exif_transpose(img)
        col_l, col_img, col_r = st.columns([1, 2, 1])
        with col_img:
            st.image(img, width="stretch")
    except Exception:
        # Malformed file that slipped past the type filter - reset cleanly.
        st.error("Could not read the uploaded image. Please try a different file.")
        st.session_state.image_bytes = None
        st.session_state.result = None
        st.session_state.pending = False

# ---------------------------------------------------------------------------
# Classify if a new image is pending. The image is already on screen (above),
# so the user sees it together with the spinner while the model runs.
# ---------------------------------------------------------------------------

if st.session_state.pending:
    # Track success outside the try block so we can call st.rerun() AFTER the
    # try/except completes. st.rerun() works by raising an internal
    # RerunException to halt the script - if called inside the try block, our
    # generic `except Exception` catches it and surfaces it as
    # "Unexpected error: RerunData(...)".
    api_call_succeeded = False

    with st.spinner("Classifying..."):
        try:
            st.session_state.result = call_api(st.session_state.image_bytes)
            api_call_succeeded = True
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the API at: " + API_URL +
                ". Make sure the API server is running."
            )
        except requests.exceptions.Timeout:
            st.error("API request timed out. The model may be loading (cold start). Try again.")
        except requests.exceptions.HTTPError as e:
            st.error("API error: " + str(e))
        except Exception as e:
            st.error("Unexpected error: " + str(e))

    st.session_state.pending = False

    if api_call_succeeded:
        # Bump the uploader key so the file_uploader widget is re-created empty
        # on the next rerun. The image and result live in session_state, so
        # they stay visible on the page.
        st.session_state.uploader_key += 1
        st.rerun()

# ---------------------------------------------------------------------------
# Display result if available
# ---------------------------------------------------------------------------

if st.session_state.result is not None:
    result = st.session_state.result

    st.markdown("<br>", unsafe_allow_html=True)

    # Winner info box
    winner_class = CLASS_LABELS.get(result["class"], result["class"].capitalize())
    winner_conf  = result["confidence"] * 100

    st.markdown(f"""
    <div class="winner-box">
        <div class="winner-label">Classified as</div>
        <div class="winner-class">{winner_class}</div>
        <div class="winner-conf">Confidence: {winner_conf:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Bar chart
    render_bar_chart(result["all_scores"])

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="footer">&copy; 2026 Jerzy Batygolski</div>',
    unsafe_allow_html=True,
)
