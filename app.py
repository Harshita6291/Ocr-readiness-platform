"""
OCR Readiness Evaluation Platform
SNLP Department — Team: Yash (Lead), Vivek, Mansi, Krish, Tanusha
Run: streamlit run app.py
"""

import sys, os
import copy
import hashlib
import io
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import cv2

from factors import run_all_factors, generate_recommendations, DISPLAY_NAMES, WEIGHTS
from factor_info import FACTOR_INFO
from storage import save_result, load_results, compute_correlations
from report import generate_pdf_report, generate_batch_pdf_report
from descriptions import get_factor_description
from short_descriptions import get_short_description
from api_integration import call_all_team_apis, KNOWN_ISSUES, get_current_urls
from config_manager import load_config, save_config, build_urls, PORTS, ENDPOINTS
from overlay import (
    get_blur_overlay, get_noise_overlay, get_skew_overlay,
    get_text_density_overlay, get_stroke_width_overlay,
    get_connected_components_overlay
)

from refinement import (
    FACTOR_KEYS as REFINEMENT_FACTOR_KEYS,
    REFINEMENT_INFO,
    RefinementCandidate,
    assess_candidate,
    bgr_to_rgb,
    evaluate_factor_refinement,
    refinement_eligibility,
    refine_all,
    run_all_factors as refinement_local_scorer,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import shutil

try:
    import pytesseract

    def _find_tesseract_cmd():
        # 1. Check environment variable
        env_cmd = os.environ.get("TESSERACT_CMD")
        if env_cmd and os.path.isfile(env_cmd):
            return env_cmd
        
        # 2. Check system PATH
        path_cmd = shutil.which("tesseract")
        if path_cmd:
            return path_cmd

        # 3. Check project-relative paths
        local_candidates = [
            os.path.join(BASE_DIR, "tesseract", "tesseract.exe"),
            os.path.join(BASE_DIR, "tesseract.exe"),
            os.path.join(BASE_DIR, "bin", "tesseract.exe"),
        ]
        for cand in local_candidates:
            if os.path.isfile(cand):
                return cand

        # 4. Check common Windows, Linux, and macOS locations
        user_profile = os.environ.get("USERPROFILE", "")
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        program_files_x86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")

        candidates = [
            os.path.join(program_files, "Tesseract-OCR", "tesseract.exe"),
            os.path.join(program_files_x86, "Tesseract-OCR", "tesseract.exe"),
            os.path.join(local_appdata, "Programs", "Tesseract-OCR", "tesseract.exe"),
            os.path.join(user_profile, "AppData", "Local", "Programs", "Tesseract-OCR", "tesseract.exe"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    def _find_tessdata_prefix():
        candidates = [
            # Check local project tessdata directory first
            os.path.join(BASE_DIR, "tessdata"),
            os.environ.get("TESSDATA_PREFIX"),
            r"C:\Program Files\Tesseract-OCR\tessdata",
            r"C:\Program Files\Tesseract-OCR",
            r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
            r"C:\Program Files (x86)\Tesseract-OCR",
            "/usr/share/tesseract-ocr/4.00/tessdata",
            "/usr/share/tesseract-ocr/5/tessdata",
            "/opt/homebrew/share/tessdata",
        ]
        for candidate in candidates:
            if not candidate or not os.path.exists(candidate):
                continue
            normalized = os.path.normpath(candidate)
            if os.path.isfile(os.path.join(normalized, "hin.traineddata")):
                return normalized
            if os.path.isfile(os.path.join(normalized, "tessdata", "hin.traineddata")):
                return os.path.join(normalized, "tessdata")

        # Secondary fallback: search for eng.traineddata
        for candidate in candidates:
            if not candidate or not os.path.exists(candidate):
                continue
            normalized = os.path.normpath(candidate)
            if os.path.isfile(os.path.join(normalized, "eng.traineddata")):
                return normalized
            if os.path.isfile(os.path.join(normalized, "tessdata", "eng.traineddata")):
                return os.path.join(normalized, "tessdata")
        return None

    tesseract_cmd = _find_tesseract_cmd()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        TESSERACT_OK = True
    else:
        TESSERACT_OK = False

    TESSDATA_PREFIX = _find_tessdata_prefix()
    if TESSDATA_PREFIX:
        os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX
        # Check if Hindi model exists
        if os.path.isfile(os.path.join(TESSDATA_PREFIX, "hin.traineddata")):
            OCR_LANG = "hin+eng"
        else:
            OCR_LANG = "eng"
    else:
        OCR_LANG = "eng"
except ImportError:
    TESSERACT_OK = False
    OCR_LANG = "eng"
    TESSDATA_PREFIX = None

try:
    from streamlit_cropper import st_cropper
    CROPPER_OK = True
except ImportError:
    CROPPER_OK = False

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Readiness Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

[data-testid="stSidebar"]{background:linear-gradient(180deg,#1A2B4A 0%,#0F1E36 100%);}
[data-testid="stSidebar"] *{color:#E8EDF5 !important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#00C4B4 !important;}

.top-banner{background:linear-gradient(90deg,#1A2B4A 0%,#0F3460 50%,#00C4B4 100%);
    padding:18px 24px;border-radius:12px;margin-bottom:20px;color:white;}
.top-banner h1{font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:700;margin:0;color:white !important;}
.top-banner p{font-size:13px;color:rgba(255,255,255,0.7);margin:4px 0 0 0;}

.metric-card{background:white;border-radius:12px;padding:16px;
    box-shadow:0 2px 8px rgba(0,0,0,0.07);border-top:3px solid #00C4B4;
    text-align:center;margin-bottom:8px;position:relative;}
.metric-card .label{font-size:13px;font-weight:700;color:#374151;
    text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;}
.metric-card .value{font-size:28px;font-weight:700;
    font-family:'Space Grotesk',sans-serif;}
.metric-card .badge{display:inline-block;padding:3px 12px;border-radius:20px;
    font-size:13px;font-weight:600;margin-top:4px;}
.badge-excellent{background:#D1FAE5;color:#065F46;}
.badge-good{background:#DBEAFE;color:#1E40AF;}
.badge-average{background:#FEF3C7;color:#92400E;}
.badge-poor{background:#FEE2E2;color:#991B1B;}
.badge-error{background:#F3F4F6;color:#6B7280;}
.metric-card .short-desc{font-size:13px;color:#4B5563;margin-top:7px;
    font-style:italic;font-weight:500;line-height:1.4;}
.metric-card .weight-src{font-size:11px;color:#9CA3AF;margin-top:5px;}

/* ── Pure CSS toggle — no JS/rerun needed ── */
.toggle-check{display:none;}
.toggle-arrow{
    display:block;margin:10px auto 0 auto;
    width:28px;height:28px;line-height:28px;text-align:center;
    background:#F3F4F6;border-radius:50%;cursor:pointer;
    font-size:14px;color:#6B7280;user-select:none;
    transition:background 0.2s,transform 0.2s;}
.toggle-arrow:hover{background:#E5E7EB;color:#374151;}
.card-info-panel{
    display:none;
    background:#EFF6FF;border-left:3px solid #3B82F6;
    border-radius:8px;padding:10px 12px;margin-top:10px;
    text-align:left;font-size:11px;color:#1E3A5F;line-height:1.7;
    overflow:hidden;word-wrap:break-word;word-break:break-word;
    white-space:normal;max-width:100%;box-sizing:border-box;}
.card-info-panel .info-title{font-weight:700;font-size:12px;
    color:#1A2B4A;margin-bottom:6px;}
.card-info-panel .info-row{margin-bottom:5px;word-wrap:break-word;
    overflow-wrap:break-word;white-space:normal;}
.card-info-panel .info-label{font-weight:600;color:#3B82F6;}
/* When checkbox is checked: show panel and rotate arrow */
.toggle-check:checked ~ .card-info-panel{display:block;}
.toggle-check:checked ~ .toggle-arrow{
    background:#DBEAFE;color:#1D4ED8;transform:rotate(180deg);}

.score-ring-wrap{display:flex;flex-direction:column;align-items:center;
    justify-content:center;padding:20px;
    background:linear-gradient(135deg,#1A2B4A,#0F3460);
    border-radius:16px;color:white;height:100%;}
.score-ring-label{font-size:12px;font-weight:600;letter-spacing:0.1em;
    text-transform:uppercase;color:#00C4B4;margin-bottom:4px;}
.score-ring-number{font-size:52px;font-weight:700;
    font-family:'Space Grotesk',sans-serif;line-height:1;}
.score-ring-status{font-size:16px;font-weight:500;color:#A8B8D0;margin-top:4px;}

.rec-box{background:#F0FDF4;border-left:4px solid #10B981;
    padding:10px 14px;border-radius:0 8px 8px 0;
    margin-bottom:8px;font-size:14px;color:#1F2937;}
.rec-box.warn{background:#FFFBEB;border-left-color:#F59E0B;}

.info-card{background:#F9FAFB;border:1px solid #E5E7EB;
    border-radius:10px;padding:16px;margin-bottom:12px;}
.info-card h4{color:#1A2B4A;margin:0 0 8px 0;
    font-family:'Space Grotesk',sans-serif;}
.owner-tag{display:inline-block;background:#EFF6FF;color:#1D4ED8;
    font-size:11px;font-weight:600;padding:2px 8px;
    border-radius:20px;margin-bottom:8px;}

.api-row{padding:6px 12px;border-radius:6px;margin-bottom:4px;font-size:12px;font-weight:500;}
.api-yash{background:#D1FAE5;color:#065F46;}
.api-mansi{background:#EDE9FE;color:#5B21B6;}
.api-krish{background:#DBEAFE;color:#1E40AF;}
.api-vivek{background:#FFEDD5;color:#9A3412;}
.api-tanusha{background:#FCE7F3;color:#9D174D;}
.api-local{background:#F3F4F6;color:#374151;}

.desc-box{background:#F8FAFC;border-left:3px solid #00C4B4;
    padding:12px 16px;border-radius:0 8px 8px 0;
    font-size:13px;color:#374151;line-height:1.6;margin-top:8px;}

.clear-btn{text-align:right;margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def pil_to_bgr(img):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def score_color(s):
    if s >= 81: return "#10B981"
    if s >= 61: return "#3B82F6"
    if s >= 41: return "#F59E0B"
    return "#EF4444"

def detect_language_stats(text: str) -> dict:
    """
    Auto-detects if the text is Hindi, English, or mixed,
    and returns character breakdown percentages.
    """
    if not text or not text.strip():
        return {
            "detected_lang": "None Detected",
            "hindi_pct": 0.0,
            "english_pct": 0.0,
            "other_pct": 0.0,
            "hindi_count": 0,
            "english_count": 0,
            "other_count": 0,
            "total_letters": 0
        }

    hindi_count = 0
    english_count = 0
    other_count = 0

    for ch in text:
        if '\u0900' <= ch <= '\u097F':
            hindi_count += 1
        elif ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'):
            english_count += 1
        elif not ch.isspace():
            other_count += 1

    total_letters = hindi_count + english_count
    if total_letters > 0:
        h_pct = round((hindi_count / total_letters) * 100, 1)
        e_pct = round((english_count / total_letters) * 100, 1)
    else:
        h_pct = 0.0
        e_pct = 0.0

    if h_pct >= 75.0:
        detected = "Hindi (हिन्दी) 🇮🇳"
    elif e_pct >= 75.0:
        detected = "English 🇬🇧"
    elif h_pct > 0 and e_pct > 0:
        detected = "Mixed (Hindi & English) 🌐"
    elif other_count > 0:
        detected = "Symbols / Numeric 🔣"
    else:
        detected = "Unknown"

    return {
        "detected_lang": detected,
        "hindi_pct": h_pct,
        "english_pct": e_pct,
        "hindi_count": hindi_count,
        "english_count": english_count,
        "other_count": other_count,
        "total_letters": total_letters
    }

def generate_word_confidence_html(word_details: list) -> str:
    """
    Generates HTML with color-coded word tags based on Tesseract confidence score.
    Green = high (>=80%), Yellow = medium (50-79%), Red = low (<50%).
    """
    if not word_details:
        return "<div style='padding:20px; text-align:center; color:#6B7280; font-style:italic;'>No word confidence data available.</div>"

    html_parts = []
    current_line = -1

    import html
    for item in word_details:
        w = item["word"].strip()
        if not w:
            continue
        c = item["conf"]
        line = item.get("line", 0)

        if current_line != -1 and line != current_line:
            html_parts.append("<br style='margin-bottom: 6px;' />")

        current_line = line

        if c >= 80:
            bg, fg, border = "#D1FAE5", "#065F46", "#A7F3D0"  # Green
        elif c >= 50:
            bg, fg, border = "#FEF3C7", "#92400E", "#FDE68A"  # Yellow
        else:
            bg, fg, border = "#FEE2E2", "#991B1B", "#FCA5A5"  # Red

        escaped_w = html.escape(w)
        span = (
            f'<span style="background-color:{bg}; color:{fg}; border:1px solid {border}; '
            f'padding:3px 8px; border-radius:6px; font-weight:600; font-family:sans-serif; '
            f'font-size:14px; margin:2px 3px; display:inline-block; transition:all 0.15s ease-in-out;" '
            f'title="Word: &quot;{escaped_w}&quot; | Confidence: {c:.1f}%">{escaped_w}'
            f'<sup style="font-size:9px; margin-left:3px; opacity:0.8;">{int(c)}%</sup></span>'
        )
        html_parts.append(span)

    return (
        '<div style="background:#F9FAFB; border:1px solid #E5E7EB; border-radius:10px; '
        'padding:16px; max-height:420px; overflow-y:auto; line-height:2.4;">'
        + "".join(html_parts) +
        '</div>'
    )

def generate_ocr_error_heatmap(bgr_img: np.ndarray, word_details: list) -> np.ndarray:
    """
    Overlays bounding boxes and a smooth jet error heatmap on the document image
    highlighting regions where Tesseract had low confidence (<50%).
    """
    img_out = bgr_img.copy()
    h_img, w_img = img_out.shape[:2]

    heatmap_mask = np.zeros((h_img, w_img), dtype=np.float32)
    has_low_conf = False

    for item in word_details:
        w = item["word"].strip()
        if not w:
            continue
        c = item["conf"]
        x, y, w_box, h_box = item["box"]

        x, y = max(0, x), max(0, y)
        w_box, h_box = max(1, w_box), max(1, h_box)
        x2, y2 = min(w_img, x + w_box), min(h_img, y + h_box)

        if c < 50:
            intensity = (50.0 - c) / 50.0
            heatmap_mask[y:y2, x:x2] = np.maximum(heatmap_mask[y:y2, x:x2], intensity)
            has_low_conf = True
            cv2.rectangle(img_out, (x, y), (x2, y2), (0, 0, 235), 2)
        elif c < 80:
            cv2.rectangle(img_out, (x, y), (x2, y2), (0, 180, 255), 1)
        else:
            cv2.rectangle(img_out, (x, y), (x2, y2), (0, 180, 0), 1)

    if has_low_conf:
        ksize = max(15, (min(h_img, w_img) // 25) | 1)
        blurred_mask = cv2.GaussianBlur(heatmap_mask, (ksize, ksize), 0)
        mask_uint8 = np.clip(blurred_mask * 255, 0, 255).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)

        alpha_channel = blurred_mask[:, :, np.newaxis] * 0.45
        img_out = (img_out * (1.0 - alpha_channel) + color_heatmap * alpha_channel).astype(np.uint8)

    return img_out

def render_copy_to_clipboard_button(text_to_copy: str, button_id: str = "copy_btn"):
    """
    Renders an interactive JS Copy to Clipboard button for Streamlit.
    """
    import html
    escaped_text = html.escape(text_to_copy).replace("\n", "\\n").replace("'", "\\'").replace('"', '&quot;')

    html_code = f"""
    <div style="margin-bottom: 12px;">
      <button id="{button_id}" onclick="copyText()" style="
        background: linear-gradient(135deg, #00C4B4 0%, #0F3460 100%);
        color: white;
        border: none;
        padding: 9px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        transition: transform 0.1s ease;
      ">
        📋 Copy Extracted Text to Clipboard
      </button>
      <span id="{button_id}_status" style="margin-left: 12px; font-size: 13px; color: #10B981; font-weight: 600; display: none;">
        ✓ Copied to clipboard!
      </span>
    </div>
    <script>
    function copyText() {{
      const text = `{escaped_text}`;
      navigator.clipboard.writeText(text).then(function() {{
        const status = document.getElementById('{button_id}_status');
        status.style.display = 'inline';
        setTimeout(() => {{ status.style.display = 'none'; }}, 2500);
      }}).catch(function(err) {{
        console.error('Failed to copy: ', err);
      }});
    }}
    </script>
    """
    st.components.v1.html(html_code, height=45)

def run_tesseract(img):
    empty_stats = detect_language_stats("")
    if not TESSERACT_OK:
        return None, "", [], None, empty_stats

    tess_config = f'--tessdata-dir "{TESSDATA_PREFIX}"' if (TESSDATA_PREFIX and os.path.exists(TESSDATA_PREFIX)) else ""

    langs_to_try = [OCR_LANG]
    if OCR_LANG != "eng":
        langs_to_try.append("eng")

    bgr_img = pil_to_bgr(img)

    for lang in langs_to_try:
        try:
            data = pytesseract.image_to_data(
                img,
                lang=lang,
                config=tess_config,
                output_type=pytesseract.Output.DICT
            )
            confs = []
            word_details = []
            n_boxes = len(data.get("text", []))

            for i in range(n_boxes):
                w = data["text"][i]
                c_raw = data["conf"][i]
                try:
                    c_val = float(c_raw)
                except (TypeError, ValueError):
                    c_val = -1.0

                if c_val >= 0:
                    confs.append(c_val)
                    if w and w.strip():
                        word_details.append({
                            "word": w,
                            "conf": c_val,
                            "box": (data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
                            "block": data.get("block_num", [0]*n_boxes)[i],
                            "line": data.get("line_num", [0]*n_boxes)[i],
                        })

            text = pytesseract.image_to_string(img, lang=lang, config=tess_config).strip()
            mean_conf = round(float(np.mean(confs)), 1) if confs else 0.0
            lang_stats = detect_language_stats(text)
            heatmap_img = generate_ocr_error_heatmap(bgr_img, word_details)
            return mean_conf, text, word_details, heatmap_img, lang_stats
        except Exception:
            if tess_config:
                try:
                    data = pytesseract.image_to_data(
                        img,
                        lang=lang,
                        output_type=pytesseract.Output.DICT
                    )
                    confs = []
                    word_details = []
                    n_boxes = len(data.get("text", []))
                    for i in range(n_boxes):
                        w = data["text"][i]
                        c_raw = data["conf"][i]
                        try:
                            c_val = float(c_raw)
                        except (TypeError, ValueError):
                            c_val = -1.0

                        if c_val >= 0:
                            confs.append(c_val)
                            if w and w.strip():
                                word_details.append({
                                    "word": w,
                                    "conf": c_val,
                                    "box": (data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
                                    "block": data.get("block_num", [0]*n_boxes)[i],
                                    "line": data.get("line_num", [0]*n_boxes)[i],
                                })

                    text = pytesseract.image_to_string(img, lang=lang).strip()
                    mean_conf = round(float(np.mean(confs)), 1) if confs else 0.0
                    lang_stats = detect_language_stats(text)
                    heatmap_img = generate_ocr_error_heatmap(bgr_img, word_details)
                    return mean_conf, text, word_details, heatmap_img, lang_stats
                except Exception:
                    pass
            continue

    return None, "", [], None, empty_stats


def make_radar(factor_results):
    keys   = list(DISPLAY_NAMES.keys())
    labels = [DISPLAY_NAMES[k] for k in keys]
    scores = [factor_results.get(k,{}).get("score",0) for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores+[scores[0]], theta=labels+[labels[0]],
        fill="toself", fillcolor="rgba(0,196,180,0.15)",
        line=dict(color="#00C4B4",width=2), name="Your Image",
        marker=dict(color="#1A2B4A",size=7),
    ))
    fig.add_trace(go.Scatterpolar(
        r=[70]*len(labels)+[70], theta=labels+[labels[0]],
        fill="none",
        line=dict(color="rgba(245,158,11,0.5)",width=1.5,dash="dot"),
        name="Good threshold (70)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#F9FAFB",
            radialaxis=dict(range=[0,100],tickfont=dict(size=9),
                gridcolor="#E5E7EB",linecolor="#D1D5DB"),
            angularaxis=dict(tickfont=dict(size=11,color="#374151"),
                linecolor="#E5E7EB"),
        ),
        showlegend=True,
        legend=dict(orientation="h",y=-0.15,font=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20,b=20,l=40,r=40),
        height=430,
    )
    return fig

# ── Refinement state and helpers ─────────────────────────────────────────────
# Refinement never changes factor formulas. It only supplies a new active image
# to the existing analysis pipeline and keeps every prior analysis reversible.
_REFINEMENT_DEFAULTS = {
    "refinement_source_signature": "",
    "refinement_preview_open": False,
    "refinement_preview_kind": "",
    "refinement_candidate_img": None,
    "refinement_candidate_results": {},
    "refinement_candidate_api_status": {},
    "refinement_baseline_img": None,
    "refinement_baseline_results": {},
    "refinement_factor": "",
    "refinement_method": "",
    "refinement_parameters": {},
    "refinement_candidate_safety": "",
    "refinement_candidate_evaluated": [],
    "refinement_history": [],
    "refinement_apply_count": 0,
    "refinement_last_applied": False,
    "refinement_info_factor": "",
    "refinement_feedback": "",
    "refinement_all_steps": [],
    "refinement_request": "",
}


def _pil_signature(image):
    """Cheap stable signature used to detect a genuinely new crop/source image."""
    if image is None:
        return ""
    preview = image.convert("RGB").copy()
    preview.thumbnail((128, 128))
    payload = f"{image.size}:{preview.size}:".encode() + preview.tobytes()
    return hashlib.sha1(payload).hexdigest()


def _clear_refinement_preview():
    st.session_state.refinement_preview_open = False
    st.session_state.refinement_preview_kind = ""
    st.session_state.refinement_candidate_img = None
    st.session_state.refinement_candidate_results = {}
    st.session_state.refinement_candidate_api_status = {}
    st.session_state.refinement_baseline_img = None
    st.session_state.refinement_baseline_results = {}
    st.session_state.refinement_factor = ""
    st.session_state.refinement_method = ""
    st.session_state.refinement_parameters = {}
    st.session_state.refinement_candidate_safety = ""
    st.session_state.refinement_candidate_evaluated = []
    st.session_state.refinement_all_steps = []


def _reset_refinement_state():
    for key, value in _REFINEMENT_DEFAULTS.items():
        st.session_state[key] = copy.deepcopy(value)


def _copy_analysis_snapshot():
    image = st.session_state.get("analysis_img")
    heatmap = st.session_state.get("ocr_heatmap_img")
    return {
        "analysis_img": image.copy() if image is not None else None,
        "final_results": copy.deepcopy(st.session_state.get("final_results", {})),
        "api_status": copy.deepcopy(st.session_state.get("api_status", {})),
        "ocr_conf": st.session_state.get("ocr_conf"),
        "ocr_text": st.session_state.get("ocr_text", ""),
        "ocr_word_details": copy.deepcopy(st.session_state.get("ocr_word_details", [])),
        "ocr_heatmap_img": heatmap.copy() if heatmap is not None else None,
        "ocr_lang_stats": copy.deepcopy(st.session_state.get("ocr_lang_stats", {})),
        "recs": copy.deepcopy(st.session_state.get("recs", [])),
    }


def _restore_analysis_snapshot(snapshot):
    for key, value in snapshot.items():
        st.session_state[key] = value
    st.session_state.analysis_done = bool(snapshot.get("final_results"))


def _score_image_with_existing_pipeline(image, use_apis):
    """The normal app scoring path, retained for applied refinements too."""
    bgr = pil_to_bgr(image)
    local_results = run_all_factors(bgr)
    if use_apis:
        return call_all_team_apis(bgr, local_results)
    return local_results, {}


def _complete_analysis_for_active_image(image, use_apis):
    """Run the same scorer, API option, Tesseract, and recommendation flow as Analyse Image."""
    final_results, api_status = _score_image_with_existing_pipeline(image, use_apis)
    recs = generate_recommendations(final_results)
    ocr_conf, ocr_text, word_details, heatmap_img, lang_stats = run_tesseract(image)
    return final_results, api_status, recs, ocr_conf, ocr_text, word_details, heatmap_img, lang_stats


def _store_refinement_preview(candidate, baseline_image, baseline_results, api_status, kind="single", all_steps=None):
    candidate_rgb = bgr_to_rgb(candidate.image)
    st.session_state.refinement_preview_open = True
    st.session_state.refinement_preview_kind = kind
    st.session_state.refinement_candidate_img = Image.fromarray(candidate_rgb)
    st.session_state.refinement_candidate_results = copy.deepcopy(candidate.scores or {})
    st.session_state.refinement_candidate_api_status = copy.deepcopy(api_status)
    st.session_state.refinement_baseline_img = baseline_image.copy()
    st.session_state.refinement_baseline_results = copy.deepcopy(baseline_results)
    st.session_state.refinement_factor = candidate.target_factor
    st.session_state.refinement_method = candidate.method
    st.session_state.refinement_parameters = copy.deepcopy(candidate.parameters)
    st.session_state.refinement_candidate_safety = candidate.safety_reason
    st.session_state.refinement_all_steps = copy.deepcopy(all_steps or [])


def _factor_info_markdown(key):
    existing = FACTOR_INFO.get(key, {})
    method = REFINEMENT_INFO.get(key, {})
    formula = existing.get("formula", "See the existing factor implementation.")
    definition = existing.get("definition", "This OCR quality factor is evaluated by the existing scorer.")
    impact = existing.get("ocr_impact", "It affects OCR reliability.")
    return (
        f"**Measures:** {definition}\n\n"
        f"**Existing scoring method:** `{formula}`\n\n"
        f"**Why it matters:** {impact}\n\n"
        f"**Refinement approach:** {method.get('details', 'A conservative image-only candidate search.')}\n\n"
        "**Safety limit:** a candidate is shown only when its target improves by at least "
        "+0.5, no other factor falls by more than 5 points, and OCR Readiness strictly improves."
    )


# ── Session state init ────────────────────────────────────────────────────────
# This keeps the image and results alive when user switches pages
if "analysis_done"    not in st.session_state: st.session_state.analysis_done    = False
if "final_results"    not in st.session_state: st.session_state.final_results    = {}
if "api_status"       not in st.session_state: st.session_state.api_status       = {}
if "ocr_conf"         not in st.session_state: st.session_state.ocr_conf         = None
if "ocr_text"         not in st.session_state: st.session_state.ocr_text         = ""
if "ocr_word_details" not in st.session_state: st.session_state.ocr_word_details = []
if "ocr_heatmap_img"  not in st.session_state: st.session_state.ocr_heatmap_img  = None
if "ocr_lang_stats"   not in st.session_state: st.session_state.ocr_lang_stats   = {}
if "image_name"       not in st.session_state: st.session_state.image_name       = ""
if "raw_pil"          not in st.session_state: st.session_state.raw_pil          = None
if "analysis_img"     not in st.session_state: st.session_state.analysis_img     = None
if "recs"             not in st.session_state: st.session_state.recs             = []
if "card_info_open"   not in st.session_state: st.session_state.card_info_open   = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
for _refinement_key, _refinement_value in _REFINEMENT_DEFAULTS.items():
    if _refinement_key not in st.session_state:
        st.session_state[_refinement_key] = copy.deepcopy(_refinement_value)

with st.sidebar:
    st.markdown("## 🔍 OCR Readiness")
    st.markdown("**SNLP Department**")
    st.markdown("---")

    nav = st.radio("Navigation", [
        "🏠 Analyse Image",
        "📊 History & Correlation",
        "📖 About Factors",
        "🔌 API Status",
        "⚙️ Settings",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Score Scale**")
    for lbl,rng,col in [
        ("Excellent","81–100","#10B981"),
        ("Good","61–80","#3B82F6"),
        ("Average","41–60","#F59E0B"),
        ("Poor","0–40","#EF4444"),
    ]:
        st.markdown(
            f'<span style="color:{col};font-weight:600;">■</span> {lbl} ({rng})',
            unsafe_allow_html=True)

    # Show status of current analysis in sidebar
    if st.session_state.analysis_done:
        st.markdown("---")
        st.markdown("**Last Analysis**")
        sc = st.session_state.final_results.get("ocr_readiness_score", 0)
        st.markdown(
            f'<span style="color:{score_color(sc)};font-size:22px;font-weight:700;">{sc}</span> '
            f'<span style="color:#A8B8D0;font-size:12px;">/ 100</span>',
            unsafe_allow_html=True)
        st.caption(f"📄 {st.session_state.image_name}")
        if st.button("🗑️ Clear Analysis", width="stretch"):
            for key in ["analysis_done","final_results","api_status",
                        "ocr_conf","ocr_text","ocr_word_details","ocr_heatmap_img","ocr_lang_stats",
                        "image_name","raw_pil","analysis_img","recs"]:
                st.session_state[key] = False if key=="analysis_done" else ({} if ("results" in key or "status" in key or "lang" in key) else (None if key in ["ocr_conf","raw_pil","analysis_img","ocr_heatmap_img"] else ([] if (key=="recs" or "word" in key) else "")))
            st.rerun()
            _reset_refinement_state()


# ════════════════════════════════════════════════
# PAGE 1 — Analyse Image
# ════════════════════════════════════════════════
if "🏠 Analyse Image" in nav:

    st.markdown("""
    <div class="top-banner">
      <h1>🔍 OCR Readiness Evaluation Platform</h1>
      <p>Upload → Crop (optional) → 10 Factor Scores → OCR Readiness Score → Tesseract Validation → PDF Report</p>
    </div>""", unsafe_allow_html=True)

    # ── Upload ──────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload document image(s) — select multiple files to enable Batch Comparison Mode",
        type=["png","jpg","jpeg","bmp","tiff","webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > 1:
            st.success(f"📦 Batch mode active — **{len(uploaded_files)} images uploaded**. Click below to evaluate all images in parallel.")
            
            use_apis = st.checkbox(
                "🔌 Use team APIs (Vivek · Mansi · Krish · Tanusha)",
                value=True,
                key="batch_api_chk"
            )

            if st.button(f"🚀 Analyse Batch ({len(uploaded_files)} Images)", type="primary", width="stretch"):
                batch_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, file_obj in enumerate(uploaded_files):
                    status_text.text(f"Processing ({idx+1}/{len(uploaded_files)}): {file_obj.name}...")
                    img_pil = Image.open(file_obj).convert("RGB")
                    bgr = pil_to_bgr(img_pil)
                    
                    local_res = run_all_factors(bgr)
                    final_res = local_res
                    api_stat = {}
                    if use_apis:
                        final_res, api_stat = call_all_team_apis(bgr, local_res)
                    
                    recs = generate_recommendations(final_res)
                    conf, txt, word_details, heatmap_img, lang_stats = run_tesseract(img_pil)
                    readiness = final_res["ocr_readiness_score"]
                    save_result(file_obj.name, final_res, readiness, conf)
                    
                    batch_results.append({
                        "image_name": file_obj.name,
                        "raw_pil": img_pil,
                        "results": final_res,
                        "api_status": api_stat,
                        "ocr_conf": conf,
                        "ocr_text": txt,
                        "word_details": word_details,
                        "heatmap_img": heatmap_img,
                        "lang_stats": lang_stats,
                        "recs": recs
                    })
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.success(f"✅ Successfully processed {len(batch_results)} images!")
                st.session_state.batch_results = batch_results
                st.session_state.batch_done = True

            # ── Render Batch Dashboard ──
            if st.session_state.get("batch_done") and st.session_state.get("batch_results"):
                b_results = st.session_state.batch_results
                st.markdown("---")
                st.markdown("### 📊 Batch Comparison Dashboard")

                # Summary metrics
                scores = [item["results"]["ocr_readiness_score"] for item in b_results]
                best_item = max(b_results, key=lambda x: x["results"]["ocr_readiness_score"])
                worst_item = min(b_results, key=lambda x: x["results"]["ocr_readiness_score"])
                avg_score = round(float(np.mean(scores)), 1)

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Total Images", len(b_results))
                mc2.metric("Batch Average", f"{avg_score} / 100")
                mc3.metric("🏆 Best Performing", f"{best_item['image_name']}", f"{best_item['results']['ocr_readiness_score']}/100")
                mc4.metric("⚠️ Worst Performing", f"{worst_item['image_name']}", f"{worst_item['results']['ocr_readiness_score']}/100")

                # Build DataFrame
                table_rows = []
                for item in b_results:
                    row = {
                        "Image Name": item["image_name"],
                        "OCR Readiness Score": item["results"]["ocr_readiness_score"],
                        "Status": item["results"]["ocr_readiness_status"],
                        "OCR Confidence": f"{item['ocr_conf']}%" if item["ocr_conf"] is not None else "N/A"
                    }
                    for fk, display in DISPLAY_NAMES.items():
                        row[display] = item["results"].get(fk, {}).get("score", 0)
                    table_rows.append(row)

                df_batch = pd.DataFrame(table_rows)

                # Highlight best and worst in table
                st.markdown("#### Score Matrix")
                factor_cols = list(DISPLAY_NAMES.values()) + ["OCR Readiness Score"]
                
                styled_df = df_batch.style.highlight_max(subset=factor_cols, props='background-color: #059669; color: #FFFFFF; font-weight: bold;') \
                                          .highlight_min(subset=factor_cols, props='background-color: #DC2626; color: #FFFFFF; font-weight: bold;')
                
                st.dataframe(styled_df, use_container_width=True)
                st.caption("💡 **Emerald Green highlight** = Best score per factor | **Ruby Red highlight** = Worst score per factor")

                # Batch PDF Download
                st.markdown("---")
                st.markdown("#### 📄 Export Combined Batch Report")
                batch_pdf = generate_batch_pdf_report(b_results)
                st.download_button(
                    "📄 Download Batch PDF Report (All Images)",
                    data=batch_pdf,
                    file_name="ocr_batch_report.pdf",
                    mime="application/pdf",
                    width="stretch"
                )

                # Individual breakdowns
                st.markdown("---")
                st.markdown("#### 🔍 Individual Document Breakdown")
                for item in b_results:
                    with st.expander(f"📄 {item['image_name']} — Score: {item['results']['ocr_readiness_score']}/100 ({item['results']['ocr_readiness_status']})"):
                        ic1, ic2 = st.columns([1, 2])
                        with ic1:
                            st.image(item["raw_pil"], caption=item["image_name"], use_container_width=True)
                        with ic2:
                            st.markdown(f"**OCR Confidence:** {item['ocr_conf'] if item['ocr_conf'] is not None else 'N/A'}%")
                            st.markdown("**Top Recommendations:**")
                            for rec in item["recs"][:3]:
                                st.markdown(f"- {rec}")
            st.stop()

        else:
            # Single image upload from array
            uploaded = uploaded_files[0]
            raw_pil = Image.open(uploaded).convert("RGB")
            if uploaded.name != st.session_state.image_name:
                st.session_state.raw_pil          = raw_pil
                st.session_state.image_name       = uploaded.name
                st.session_state.analysis_done     = False
                st.session_state.final_results     = {}
                st.session_state.ocr_word_details = []
                st.session_state.ocr_heatmap_img  = None
                st.session_state.ocr_lang_stats   = {}
                st.session_state.analysis_img      = raw_pil
                _reset_refinement_state()
    elif st.session_state.raw_pil is None:
        st.info("👆 Upload document image(s) to begin the analysis.")
        st.stop()

    raw_pil    = st.session_state.raw_pil
    image_name = st.session_state.image_name

    # ── Step 1: Crop ─────────────────────────────
    st.markdown("### Step 1 — Select Region")
    use_crop = st.checkbox("✂️ Crop the image before analysis")

    selected_source_img = raw_pil
    if use_crop:
        if CROPPER_OK:
            st.markdown("**Drag the handles to select the region you want to analyse:**")
            cropped = st_cropper(
                raw_pil,
                realtime_update=True,
                box_color="#00C4B4",
                aspect_ratio=None,
            )
            col_prev, col_info = st.columns([2,1])
            with col_prev:
                st.image(cropped, caption="Selected crop region", width="stretch")
            with col_info:
                w, h = cropped.size
                st.metric("Width", f"{w} px")
                st.metric("Height", f"{h} px")
            selected_source_img = cropped
        else:
            st.warning("streamlit-cropper not installed. Using slider crop instead.")
            c1, c2 = st.columns(2)
            with c1:
                st.image(raw_pil, caption="Original Image", width="stretch")
            with c2:
                w, h = raw_pil.size
                left   = st.slider("Left",   0, w-1, 0)
                top    = st.slider("Top",    0, h-1, 0)
                right  = st.slider("Right",  1, w,   w)
                bottom = st.slider("Bottom", 1, h,   h)
                if right  <= left:  right  = left  + 1
                if bottom <= top:   bottom = top   + 1
                cropped = raw_pil.crop((left, top, right, bottom))
                st.image(cropped, caption="Cropped Region", width="stretch")
            selected_source_img = cropped
    else:
        st.image(raw_pil, caption="Full image — will be analysed", width="stretch")
        selected_source_img = raw_pil

    analysis_img = st.session_state.analysis_img
    source_signature = f"{'crop' if use_crop else 'full'}:{_pil_signature(selected_source_img)}"
    if source_signature != st.session_state.refinement_source_signature:
        # A new crop/source invalidates results and any refinements based on the old pixels.
        _reset_refinement_state()
        st.session_state.refinement_source_signature = source_signature
        st.session_state.analysis_img = selected_source_img.copy()
        st.session_state.analysis_done = False
        st.session_state.final_results = {}
        st.session_state.api_status = {}
        st.session_state.ocr_conf = None
        st.session_state.ocr_text = ""
        st.session_state.ocr_word_details = []
        st.session_state.ocr_heatmap_img = None
        st.session_state.ocr_lang_stats = {}
        st.session_state.recs = []


    # ── Step 2: Analyse ──────────────────────────
    st.markdown("### Step 2 — Run Analysis")
    use_apis = st.checkbox(
        "🔌 Use team APIs (Vivek · Mansi · Krish · Tanusha) — falls back to local if any API is offline",
        value=True,
    )

    if st.button("🚀 Analyse Image", type="primary", width="stretch"):

        bgr = pil_to_bgr(analysis_img)

        with st.spinner("⚙️ Computing local factors…"):
            local_results = run_all_factors(bgr)

        final_results = local_results
        api_status    = {}

        if use_apis:
            with st.spinner("🔌 Calling team APIs (Vivek · Mansi · Krish · Tanusha)…"):
                final_results, api_status = call_all_team_apis(bgr, local_results)

        recs = generate_recommendations(final_results)

        with st.spinner("📝 Running Tesseract OCR…"):
            ocr_conf, ocr_text, ocr_word_details, ocr_heatmap_img, ocr_lang_stats = run_tesseract(analysis_img)

        ocr_readiness = final_results["ocr_readiness_score"]

        save_result(image_name, final_results, ocr_readiness, ocr_conf)

        # Store everything in session state
        st.session_state.analysis_done    = True
        st.session_state.final_results    = final_results
        st.session_state.api_status       = api_status
        st.session_state.ocr_conf         = ocr_conf
        st.session_state.ocr_text         = ocr_text or ""
        st.session_state.ocr_word_details = ocr_word_details
        st.session_state.ocr_heatmap_img  = ocr_heatmap_img
        st.session_state.ocr_lang_stats   = ocr_lang_stats
        st.session_state.recs             = recs

    # ── Show results if analysis has been done ────
    if st.session_state.analysis_done:

        final_results = st.session_state.final_results
        api_status    = st.session_state.api_status
        ocr_conf      = st.session_state.ocr_conf
        ocr_text      = st.session_state.ocr_text
        recs          = st.session_state.recs
        ocr_readiness = final_results["ocr_readiness_score"]
        ocr_stat      = final_results["ocr_readiness_status"]

        # ── API source summary ───────────────────
        if api_status:
            with st.expander("🔌 Data source for each factor"):
                for key, src in api_status.items():
                    disp = DISPLAY_NAMES.get(key, key)
                    if "Yash" in src:       css = "api-yash"
                    elif "Mansi" in src:    css = "api-mansi"
                    elif "Krish" in src:    css = "api-krish"
                    elif "Vivek" in src:    css = "api-vivek"
                    elif "Tanusha" in src:  css = "api-tanusha"
                    else:                   css = "api-local"
                    st.markdown(
                        f'<div class="api-row {css}"><b>{disp}</b>: {src}</div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Results")

        # ── 3 score rings ────────────────────────
        r1, r2, r3 = st.columns(3)

        with r1:
            c = score_color(ocr_readiness)
            st.markdown(f"""
            <div class="score-ring-wrap">
              <div class="score-ring-label">OCR Readiness Score</div>
              <div class="score-ring-number" style="color:{c};">{ocr_readiness}</div>
              <div class="score-ring-status">{ocr_stat}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            if ocr_conf is not None:
                c2 = score_color(ocr_conf)
                st.markdown(f"""
                <div class="score-ring-wrap">
                  <div class="score-ring-label">Tesseract Confidence</div>
                  <div class="score-ring-number" style="color:{c2};">{ocr_conf}%</div>
                  <div class="score-ring-status">Actual OCR</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="score-ring-wrap">
                  <div class="score-ring-label">Tesseract Confidence</div>
                  <div class="score-ring-number" style="color:#6B7280;">N/A</div>
                  <div class="score-ring-status">Not available</div>
                </div>""", unsafe_allow_html=True)

        with r3:
            if ocr_conf is not None:
                diff = ocr_readiness - ocr_conf
                dc   = "#10B981" if abs(diff)<10 else "#F59E0B" if abs(diff)<20 else "#EF4444"
                lbl  = "Predicted ≈ Actual ✓" if abs(diff)<10 else "Small deviation" if abs(diff)<20 else "Large deviation"
                st.markdown(f"""
                <div class="score-ring-wrap">
                  <div class="score-ring-label">Predicted vs Actual</div>
                  <div class="score-ring-number" style="color:{dc};">{diff:+.1f}</div>
                  <div class="score-ring-status">{lbl}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="score-ring-wrap">
                  <div class="score-ring-label">Predicted vs Actual</div>
                  <div class="score-ring-number" style="color:#6B7280;">—</div>
                  <div class="score-ring-status">Tesseract needed</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 10 factor cards ─────────────────────
        st.markdown("#### Factor Scores")
        cols = st.columns(4)
        refine_action_col, refine_status_col, refine_revert_col = st.columns([1.2, 2.4, 1.0])
        with refine_action_col:
            if st.button("✨ Refine All", key="refinement_refine_all", width="stretch"):
                st.session_state.refinement_request = "__all__"
                st.session_state.refinement_feedback = ""
        with refine_status_col:
            st.caption("Refinement changes only image pixels. Every candidate is rescored by the existing 10-factor pipeline and must improve OCR Readiness.")
        with refine_revert_col:
            if st.session_state.refinement_history and st.button("↩ Revert", key="refinement_revert", width="stretch"):
                snapshot = st.session_state.refinement_history.pop()
                _restore_analysis_snapshot(snapshot)
                st.session_state.refinement_last_applied = bool(st.session_state.refinement_history)
                _clear_refinement_preview()
                st.session_state.refinement_feedback = "Reverted to the immediately previous active image and analysis."
                st.rerun()

        if st.session_state.refinement_last_applied and st.session_state.analysis_img is not None:
            img_buffer = io.BytesIO()
            st.session_state.analysis_img.save(img_buffer, format="PNG", optimize=True)
            st.download_button(
                "⬇️ Download Refined Image",
                data=img_buffer.getvalue(),
                file_name=f"refined_{image_name.rsplit('.', 1)[0]}.png",
                mime="image/png",
                key="refinement_download",
            )
        if st.session_state.refinement_feedback:
            st.info(st.session_state.refinement_feedback)

        for i, (key, display) in enumerate(DISPLAY_NAMES.items()):
            r    = final_results.get(key, {})
            sc   = r.get("score", 0)
            s    = r.get("status", "—")
            col  = score_color(float(sc))
            bcls = f"badge-{s.lower()}"
            src_tag = ""
            if api_status:
                src = api_status.get(key, "")
                if "✅" in src:
                    src_tag = '<span style="font-size:11px;color:#065F46;font-weight:600;">● API</span>'
                else:
                    src_tag = '<span style="font-size:11px;color:#9CA3AF;">● Local</span>'

            short_desc = get_short_description(key, float(sc))

            # Build info panel content
            info_html = ""
            if key in FACTOR_INFO:
                fi    = FACTOR_INFO[key]
                ideal = fi["ideal_range"].split(".")[0]

                # Clean text — remove newlines, bullets, special chars that break HTML
                def clean_text(text, limit=150):
                    import html
                    t = text.replace("\n", " ").replace("•", "-").replace("·", "-")
                    t = html.escape(t)   # escapes <, >, &, ", '
                    t = t[:limit] + "…" if len(t) > limit else t
                    return t

                defn  = clean_text(fi["definition"], 150)
                imp   = clean_text(fi["ocr_impact"], 150)
                ideal_clean = clean_text(ideal, 100)

                info_html = f"""
                <div class="info-row"><span class="info-label">&#128100; Owner:</span> {fi['owner']}</div>
                <div class="info-row"><span class="info-label">&#128214; What it is:</span> {defn}</div>
                <div class="info-row"><span class="info-label">&#127919; OCR Impact:</span> {imp}</div>
                <div class="info-row"><span class="info-label">&#9989; Ideal Range:</span> {ideal_clean}</div>"""

            # Unique checkbox ID for each card (pure CSS toggle)
            chk_id = f"chk_{key}"

            with cols[i % 4]:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">{display}</div>
                  <div class="value" style="color:{col};">{sc}</div>
                  <span class="badge {bcls}">{s}</span>
                  <div class="short-desc">{short_desc}</div>
                  <div class="weight-src">Weight: {int(WEIGHTS[key]*100)}% &nbsp;{src_tag}</div>

                  <input type="checkbox" class="toggle-check" id="{chk_id}">
                  <div class="card-info-panel">
                    <div class="info-title">📐 {FACTOR_INFO[key]['display_name'] if key in FACTOR_INFO else display}</div>
                    {info_html}
                  </div>
                  <label class="toggle-arrow" for="{chk_id}">∨</label>
                </div>""", unsafe_allow_html=True)

                info_col, refine_col = st.columns(2)
                with info_col:
                    if st.button("ⓘ Info", key=f"refinement_info_{key}", width="stretch"):
                        st.session_state.refinement_info_factor = key
                eligibility, _eligibility_message = refinement_eligibility(float(sc))
                with refine_col:
                    if eligibility == "eligible":
                        if st.button("✨ Refine", key=f"refinement_refine_{key}", width="stretch"):
                            st.session_state.refinement_request = key
                            st.session_state.refinement_feedback = ""
                    elif eligibility == "severe":
                        st.caption("⚠️ Auto-refinement disabled — recapture/rescan required.")
                    else:
                        st.caption("✓ No refinement needed")

        st.markdown("<br>", unsafe_allow_html=True)

        info_factor = st.session_state.refinement_info_factor
        if info_factor in DISPLAY_NAMES:
            with st.expander(f"ⓘ {DISPLAY_NAMES[info_factor]} — scoring and refinement information", expanded=True):
                st.markdown(_factor_info_markdown(info_factor))
                if st.button("Close info", key="refinement_close_info"):
                    st.session_state.refinement_info_factor = ""
                    st.rerun()

        refinement_request = st.session_state.get("refinement_request", "")
        if refinement_request:
            st.session_state.refinement_request = ""
            baseline_image = analysis_img.copy()
            baseline_results = copy.deepcopy(final_results)
            baseline_bgr = pil_to_bgr(baseline_image)

            if refinement_request == "__all__":
                with st.spinner("✨ Safely evaluating eligible factors one at a time…"):
                    local_baseline = refinement_local_scorer(baseline_bgr)
                    all_outcome = refine_all(baseline_bgr, local_baseline, refinement_local_scorer)

                if not all_outcome.steps:
                    st.session_state.refinement_feedback = all_outcome.message
                else:
                    all_candidate = RefinementCandidate(
                        all_outcome.final_image,
                        "refine_all",
                        "Safe sequential Refine All",
                        {"steps": len(all_outcome.steps)},
                    )
                    with st.spinner("Verifying the final Refine All image through the active analysis pipeline…"):
                        final_candidate_scores, final_candidate_api = _score_image_with_existing_pipeline(
                            Image.fromarray(bgr_to_rgb(all_candidate.image)), use_apis
                        )
                    all_candidate.scores = final_candidate_scores
                    all_candidate.readiness_improvement = round(
                        float(final_candidate_scores["ocr_readiness_score"]) - float(baseline_results["ocr_readiness_score"]), 1
                    )
                    all_candidate.non_target_changes = {
                        factor: round(
                            float(final_candidate_scores[factor]["score"]) - float(baseline_results[factor]["score"]), 1
                        ) for factor in REFINEMENT_FACTOR_KEYS
                    }
                    all_candidate.safe = (
                        all_candidate.readiness_improvement > 0.0
                        and all(delta >= -5.0 for delta in all_candidate.non_target_changes.values())
                    )
                    all_candidate.safety_reason = (
                        "Passed the final Refine All OCR Readiness and all-factor safety checks."
                        if all_candidate.safe else "No safe refinement was found for this factor set."
                    )
                    if all_candidate.safe:
                        all_steps = [
                            {"factor": DISPLAY_NAMES[step.target_factor], "method": step.method,
                             "ocr_readiness_change": step.readiness_improvement}
                            for step in all_outcome.steps
                        ]
                        _store_refinement_preview(
                            all_candidate, baseline_image, baseline_results, final_candidate_api,
                            kind="all", all_steps=all_steps,
                        )
                    else:
                        st.session_state.refinement_feedback = "No safe refinement was found for the eligible factors."

            elif refinement_request in DISPLAY_NAMES:
                shown_score = float(baseline_results.get(refinement_request, {}).get("score", 0.0))
                eligibility, eligibility_message = refinement_eligibility(shown_score)
                if eligibility == "severe":
                    st.session_state.refinement_feedback = eligibility_message
                elif eligibility == "good":
                    st.session_state.refinement_feedback = eligibility_message
                else:
                    with st.spinner(f"✨ Generating and scoring {DISPLAY_NAMES[refinement_request]} candidates…"):
                        # Every candidate is measured by the unmodified local 10-factor scorer.
                        # The chosen local candidate is then verified through the currently active
                        # app pipeline (including optional team APIs) before it can be previewed.
                        local_baseline = refinement_local_scorer(baseline_bgr)
                        outcome = evaluate_factor_refinement(
                            baseline_bgr, refinement_request, local_baseline, refinement_local_scorer
                        )
                    candidate = outcome.best_candidate
                    if candidate is None:
                        st.session_state.refinement_feedback = outcome.message
                    else:
                        with st.spinner("Verifying the selected candidate through the active analysis pipeline…"):
                            final_candidate_scores, final_candidate_api = _score_image_with_existing_pipeline(
                                Image.fromarray(bgr_to_rgb(candidate.image)), use_apis
                            )
                        assess_candidate(candidate, baseline_results, final_candidate_scores)
                        if candidate.safe:
                            _store_refinement_preview(
                                candidate, baseline_image, baseline_results, final_candidate_api
                            )
                            st.session_state.refinement_candidate_evaluated = outcome.evaluated
                        else:
                            st.session_state.refinement_feedback = "No safe refinement was found for this factor."

        # ── Tabs ─────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 Factor Details",
            "📝 Extracted Text Quality",
            "💡 Recommendations",
            "🔍 Visual Overlays",
        ])

        with tab1:
            st.markdown("Click any factor to see its detailed explanation:")
            for key, display in DISPLAY_NAMES.items():
                r      = final_results.get(key, {})
                sc     = r.get("score", 0)
                st_txt = r.get("status","—")
                src    = api_status.get(key,"local") if api_status else "local"
                col    = score_color(float(sc))
                # Get rich description based on score range
                rich_desc = get_factor_description(key, float(sc))
                with st.expander(f"**{display}** — {sc}/100   ({st_txt})"):
                    ca, cb = st.columns([3,1])
                    with ca:
                        # Rich description box
                        st.markdown(
                            f'<div class="desc-box">{rich_desc}</div>',
                            unsafe_allow_html=True)
                        if key == "stroke_width_score":
                            details = r.get("details", {})
                            sw_px = details.get("median_stroke_width_px")
                            ch_px = details.get("median_char_height_px")
                            ratio = details.get("stroke_to_height_ratio", r.get("raw_value"))
                            if sw_px is not None and ch_px is not None and ch_px > 0:
                                st.caption(
                                    f"📊 **Stroke Width:** `{sw_px:.2f} px` &nbsp;|&nbsp; "
                                    f"**Character Height:** `{ch_px:.1f} px`"
                                )
                            elif r.get("raw_value") is not None:
                                st.caption(f"📊 Raw value: {r['raw_value']}  {r.get('unit','')}")
                        else:
                            if r.get("raw_value") is not None:
                                st.caption(f"📊 Raw value: {r['raw_value']}  {r.get('unit','')}")
                        st.caption(f"🔌 Source: {src}")
                    with cb:
                        st.markdown(f"<div style='text-align:center;font-size:36px;font-weight:700;color:{col};'>{sc}</div>", unsafe_allow_html=True)
                        st.progress(int(min(sc, 100)))
                        st.markdown(f"<div style='text-align:center;font-size:12px;color:#6B7280;'>out of 100</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("### 📝 Extracted Text Quality Comparison")
            if ocr_conf is not None:
                lang_stats = st.session_state.get("ocr_lang_stats", detect_language_stats(ocr_text))
                word_details = st.session_state.get("ocr_word_details", [])
                heatmap_img = st.session_state.get("ocr_heatmap_img", None)

                # ── Top Metrics Bar ──
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric("Tesseract Mean Confidence", f"{ocr_conf}%")
                with mcol2:
                    st.metric("Detected Language", lang_stats.get("detected_lang", "Unknown"))
                with mcol3:
                    h_p = lang_stats.get("hindi_pct", 0.0)
                    e_p = lang_stats.get("english_pct", 0.0)
                    st.metric("Language Breakdown", f"🇮🇳 {h_p}% Hin  |  🇬🇧 {e_p}% Eng")

                st.markdown("---")

                # ── Side-by-Side View ──
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    st.markdown("#### 🖼️ Original Image")
                    st.image(analysis_img, caption="Analyzed Image Region", use_container_width=True)

                with col_right:
                    st.markdown("#### 🔤 Extracted Text & Quality Analysis")

                    if ocr_text and ocr_text.strip():
                        render_copy_to_clipboard_button(ocr_text, "copy_btn_main")

                    view_mode = st.radio(
                        "Extracted Text Display Mode",
                        ["🎨 Word Confidence Highlighting", "🗺️ Character Error Heatmap", "📄 Plain Text"],
                        horizontal=True,
                    )

                    if "Word Confidence" in view_mode:
                        st.markdown("""
                        <div style="display:flex; gap:12px; font-size:12px; margin-bottom:8px; font-weight:600;">
                          <span style="color:#065F46; background:#D1FAE5; padding:2px 8px; border-radius:4px; border:1px solid #A7F3D0;">🟩 High (≥80%)</span>
                          <span style="color:#92400E; background:#FEF3C7; padding:2px 8px; border-radius:4px; border:1px solid #FDE68A;">🟨 Medium (50–79%)</span>
                          <span style="color:#991B1B; background:#FEE2E2; padding:2px 8px; border-radius:4px; border:1px solid #FCA5A5;">🟥 Low (<50%)</span>
                        </div>
                        """, unsafe_allow_html=True)

                        w_html = generate_word_confidence_html(word_details)
                        st.markdown(w_html, unsafe_allow_html=True)
                        st.caption("💡 Hover over any word tag to see its exact Tesseract confidence percentage.")

                    elif "Character Error Heatmap" in view_mode:
                        if heatmap_img is not None:
                            heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
                            st.image(heatmap_rgb, caption="OCR Bounding Boxes & Low-Confidence Error Heatmap", use_container_width=True)
                            st.caption("🔴 **Red Boxes/Heat**: Low confidence words (<50%) | 🟡 **Yellow**: Medium confidence (50-79%) | 🟢 **Green**: High confidence (≥80%)")
                        else:
                            st.info("No heatmap data available.")

                    elif "Plain Text" in view_mode:
                        if ocr_text and ocr_text.strip():
                            st.text_area("Extracted Plain Text", ocr_text, height=280)
                        else:
                            st.info("ℹ️ Tesseract ran successfully, but no readable text was detected in the provided image.")
            else:
                st.warning(
                    "⚠️ **Tesseract OCR executable was not automatically detected on this system.**\n\n"
                    "**To fix this on any machine:**\n"
                    "1. Install Tesseract OCR for Windows (e.g. from UB-Mannheim / GitHub) or Linux (`sudo apt install tesseract-ocr`).\n"
                    "2. Or place `tesseract.exe` into a `tesseract/` or `bin/` folder inside this project directory.\n"
                    "3. Run the automated zero-setup script (`run.bat` or `python run.py`) which automatically sets paths."
                )

        with tab3:
            for rec in recs:
                is_warn = "🔧" in rec
                box_cls = "rec-box warn" if is_warn else "rec-box"
                clean = rec.replace("**","<b>",1).replace("**","</b>",1)
                st.markdown(f'<div class="{box_cls}">{clean}</div>',
                            unsafe_allow_html=True)

        with tab4:
            st.markdown("### 🔍 Explainable Visual Overlays")
            st.caption("Visualize exactly where problems are detected on your document image:")

            overlay_choice = st.selectbox(
                "Select Factor Visual Overlay",
                [
                    "Blur Heatmap (Sharp vs Blurry regions)",
                    "Noise Map (High background noise patches)",
                    "Skew Angle Vector (Detected text baseline skew)",
                    "Text Density (Paragraph bounding boxes)",
                    "Stroke Width Map (Color coded stroke thickness)",
                    "Connected Components (Glyph bounding boxes)",
                    "OCR Error Heatmap (Tesseract Confidence Bounding Map)",
                ]
            )

            bgr_curr = pil_to_bgr(analysis_img)

            if "Blur Heatmap" in overlay_choice:
                overlay_bgr = get_blur_overlay(bgr_curr)
                st.info("🔴 **Red/Yellow**: Sharp crisp edges | 🔵 **Blue/Cyan**: Blurry or low-contrast regions")
            elif "Noise Map" in overlay_choice:
                overlay_bgr = get_noise_overlay(bgr_curr)
                st.info("🔥 **Hot Patches (Yellow/Red)**: Background speckles and high noise variance patches")
            elif "Skew Angle" in overlay_choice:
                overlay_bgr = get_skew_overlay(bgr_curr)
                st.info("📐 **Cyan Horizontal Baseline** vs **Vector Line (Green = Aligned <3°, Red = Skewed >3°)**")
            elif "Text Density" in overlay_choice:
                overlay_bgr = get_text_density_overlay(bgr_curr)
                st.info("🟩 **Teal Rectangles**: Detected paragraph blocks and text regions")
            elif "Stroke Width" in overlay_choice:
                overlay_bgr = get_stroke_width_overlay(bgr_curr)
                st.info("🟢 **Green**: Ideal stroke width (1.5-4.5px) | 🔴 **Red**: Thin stroke (<1.5px) | 🟠 **Orange**: Thick stroke (>4.5px)")
            elif "Connected Components" in overlay_choice:
                overlay_bgr = get_connected_components_overlay(bgr_curr)
                st.info("🎨 **Multi-colored Bounding Boxes**: Individual connected glyph components detected")
            elif "OCR Error Heatmap" in overlay_choice:
                heatmap_img = st.session_state.get("ocr_heatmap_img", None)
                if heatmap_img is not None:
                    overlay_bgr = heatmap_img
                    st.info("🔴 **Red Boxes/Heat**: Low confidence words (<50%) | 🟡 **Yellow**: Medium confidence (50-79%) | 🟢 **Green**: High confidence (≥80%)")
                else:
                    overlay_bgr = get_blur_overlay(bgr_curr)
                    st.info("No OCR heatmap data available yet. Displaying blur heatmap instead.")

            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, caption=overlay_choice, use_container_width=True)

        # ── PDF Export ───────────────────────────
        st.markdown("---")
        st.markdown("#### Export Report")
        pdf_bytes = generate_pdf_report(
            image_name, final_results, ocr_readiness, ocr_conf, recs
        )
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"ocr_report_{image_name.rsplit('.',1)[0]}.pdf",
            mime="application/pdf",
            width="stretch",
        )


# ════════════════════════════════════════════════
# PAGE 2 — History & Correlation
# ════════════════════════════════════════════════
elif "📊 History" in nav:

    st.markdown("""
    <div class="top-banner">
      <h1>📊 History & Correlation Analysis</h1>
      <p>All past analyses · CSV download · Factor vs OCR accuracy correlation</p>
    </div>""", unsafe_allow_html=True)

    df = load_results()
    if df is None:
        st.info("No results yet. Analyse some images first!")
        st.stop()

    st.markdown(f"**{len(df)} analyses stored in results.csv**")
    st.dataframe(df, width="stretch")

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv_bytes, "results.csv", "text/csv")

    st.markdown("---")
    st.markdown("### Factor ↔ OCR Confidence Correlation")

    corr = compute_correlations()
    if corr is None:
        st.info("Need at least 3 analyses with Tesseract data to compute correlations.")
    else:
        fig = go.Figure(go.Bar(
            x=corr.values,
            y=[DISPLAY_NAMES.get(k,k) for k in corr.index],
            orientation="h",
            marker=dict(color=[score_color(abs(v)*100) for v in corr.values]),
        ))
        fig.update_layout(
            title="Pearson Correlation with OCR Confidence",
            xaxis=dict(range=[-1,1], title="Correlation Coefficient"),
            yaxis=dict(autorange="reversed"),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Closer to +1.0 means the factor is a stronger positive predictor of OCR accuracy.")

        corr_df = pd.DataFrame({
            "Factor":      [DISPLAY_NAMES.get(k,k) for k in corr.index],
            "Correlation": [round(v,3) for v in corr.values],
            "Strength":    ["Strong" if abs(v)>0.7 else "Moderate" if abs(v)>0.4 else "Weak"
                            for v in corr.values],
        })
        st.dataframe(corr_df, width="stretch", hide_index=True)


# ════════════════════════════════════════════════
# PAGE 3 — About Factors
# ════════════════════════════════════════════════
elif "📖 About" in nav:

    st.markdown("""
    <div class="top-banner">
      <h1>📖 About the 10 Quality Factors</h1>
      <p>Definition · Importance · Formula · OCR Impact · Ideal Range</p>
    </div>""", unsafe_allow_html=True)

    selected = st.selectbox(
        "Select a Quality Factor",
        options=list(FACTOR_INFO.keys()),
        format_func=lambda k: FACTOR_INFO[k]["display_name"],
    )
    info = FACTOR_INFO[selected]

    # ── Main info card using native Streamlit ──
    with st.container(border=True):
        st.markdown(f"## 📐 {info['display_name']}")
        st.markdown(
            f'<span style="display:inline-block;background:#EFF6FF;color:#1D4ED8;'
            f'font-size:12px;font-weight:600;padding:3px 10px;border-radius:20px;'
            f'margin-bottom:12px;">👤 Assigned to: {info["owner"]}</span>',
            unsafe_allow_html=True)
        st.markdown(f"**{info['definition']}**")

    st.markdown("")

    # ── 2-column detail section ──
    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("### 📌 Why This Factor Matters")
            st.markdown(info["importance"])

        with st.container(border=True):
            st.markdown("### 🎯 Effect on OCR")
            st.markdown(info["ocr_impact"])

    with c2:
        with st.container(border=True):
            st.markdown("### 🧮 Calculation Method")
            st.code(info["formula"], language=None)

        with st.container(border=True):
            st.markdown("### ✅ Ideal Score Range")
            st.success(info["ideal_range"])

    st.markdown("---")
    st.markdown("#### All 10 Factors at a Glance")
    rows = [{
        "Factor":      v["display_name"],
        "Owner":       v["owner"],
        "Weight":      f"{int(WEIGHTS[k]*100)}%",
        "Ideal Range": v["ideal_range"].split(".")[0],
    } for k,v in FACTOR_INFO.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ════════════════════════════════════════════════
# PAGE 4 — API Status
# ════════════════════════════════════════════════
elif "🔌 API Status" in nav:

    st.markdown("""
    <div class="top-banner">
      <h1>🔌 Team API Status</h1>
      <p>Configuration · Live ping · Expected request & response formats</p>
    </div>""", unsafe_allow_html=True)

    urls = get_current_urls()
    st.markdown("### API Configuration")
    api_map = pd.DataFrame([
        {"Member":"Vivek",   "Factors":"Stroke Width, Text Density",
         "URL":urls["vivek"],   "Method":"POST", "Field":"file", "Port":"8001"},
        {"Member":"Mansi",   "Factors":"Blur, Contrast",
         "URL":urls["mansi"],   "Method":"POST", "Field":"file", "Port":"8000"},
        {"Member":"Krish",   "Factors":"Matra Continuity, Zone Integrity",
         "URL":urls["krish"],   "Method":"POST", "Field":"file", "Port":"8002"},
        {"Member":"Tanusha", "Factors":"CC Stability, Skew Penalty",
         "URL":urls["tanusha"], "Method":"POST", "Field":"file", "Port":"9001"},
        {"Member":"Yash",    "Factors":"Noise, Resolution",
         "URL":"Local (built-in)", "Method":"—", "Field":"—", "Port":"—"},
    ])
    st.dataframe(api_map, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### Live Connectivity Check")
    if st.button("🔄 Ping All APIs Now"):
        import requests as req
        tests = [
            ("Vivek",   urls["vivek"]),
            ("Mansi",   urls["mansi"]),
            ("Krish",   urls["krish"]),
            ("Tanusha", urls["tanusha"]),
        ]
        for name, url in tests:
            # Ping the actual API endpoint directly using HEAD/GET
            # Do NOT strip the port — keep full URL including port number
            try:
                r = req.get(url, timeout=5)
                st.success(f"**{name}** ({url}): Server reachable ✅")
            except req.exceptions.ConnectionError:
                st.error(f"**{name}** ({url}): Unreachable ❌ — Server not running")
            except req.exceptions.Timeout:
                st.error(f"**{name}** ({url}): Unreachable ❌ — Connection timed out")
            except Exception as e:
                # 405/422 means server IS running but endpoint needs POST not GET — that's fine!
                if hasattr(e, 'response') and e.response is not None:
                    st.success(f"**{name}** ({url}): Server reachable ✅")
                else:
                    st.error(f"**{name}** ({url}): Unreachable ❌ — {type(e).__name__}")

    st.markdown("---")
    st.markdown("### Expected Response Formats")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Vivek** `POST /analyze-image`")
        st.code('{\n  "filename": "image.png",\n  "stroke_width": 72.5,\n  "text_density": 85.1\n}', language="json")
    with c2:
        st.markdown("**Mansi** `POST /analyze`")
        st.code('{\n  "blur_score": 88.0,\n  "contrast_score": 74.3\n}', language="json")
    with c3:
        st.markdown("**Krish** `POST /scores`")
        st.code('{\n  "matra_continuity_score": 91.2,\n  "zone_integrity_score": 78.5\n}', language="json")
    with c4:
        st.markdown("**Tanusha** `POST /analyze`")
        st.code('{\n  "connected_component_stability_score": 84.0,\n  "skew_penalty_score": 91.5\n}', language="json")

    st.markdown("---")
    st.info("ℹ️ If any API is offline during analysis, the platform automatically falls back to the local algorithm for that factor.")

# ════════════════════════════════════════════════
# PAGE 5 — Settings
# ════════════════════════════════════════════════
elif "⚙️ Settings" in nav:

    st.markdown("""
    <div class="top-banner">
      <h1>⚙️ Settings — Team IP Addresses</h1>
      <p>Update your teammates' IP addresses here — no code editing needed ever again</p>
    </div>""", unsafe_allow_html=True)

    cfg = load_config()

    st.markdown("### How to find a teammate's IP address")
    st.code("ipconfig        # Windows — look for IPv4 Address\nifconfig        # Mac / Linux", language="bash")
    st.info("📌 Ask each teammate to run the command above and send you their **IPv4 Address** (looks like 192.168.x.x)")

    st.markdown("---")
    st.markdown("### Enter IP Addresses")
    st.markdown("Leave as `127.0.0.1` if that person is running on **your laptop**.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🟢 Vivek** — Port 8001 — Stroke Width & Text Density")
        vivek_ip = st.text_input("Vivek's IP Address", value=cfg["vivek_ip"],
                                  placeholder="e.g. 192.168.1.105")

        st.markdown("**🟡 Mansi** — Port 8000 — Blur & Contrast")
        mansi_ip = st.text_input("Mansi's IP Address", value=cfg["mansi_ip"],
                                  placeholder="e.g. 192.168.1.108")

    with col2:
        st.markdown("**🔵 Krish** — Port 8002 — Matra & Zone Integrity")
        krish_ip = st.text_input("Krish's IP Address", value=cfg["krish_ip"],
                                  placeholder="e.g. 192.168.1.112")

        st.markdown("**🩷 Tanusha** — Port 9001 — CC Stability & Skew")
        tanusha_ip = st.text_input("Tanusha's IP Address", value=cfg["tanusha_ip"],
                                    placeholder="e.g. 192.168.1.115")

    st.markdown("---")
    st.markdown("**Preview — URLs that will be used after saving:**")
    preview = build_urls({
        "vivek_ip": vivek_ip, "mansi_ip": mansi_ip,
        "krish_ip": krish_ip, "tanusha_ip": tanusha_ip
    })
    for name, url in preview.items():
        st.code(f"{name.capitalize()}: {url}")

    if st.button("💾 Save IP Addresses", type="primary", width="stretch"):
        new_cfg = {
            "vivek_ip":   vivek_ip.strip(),
            "mansi_ip":   mansi_ip.strip(),
            "krish_ip":   krish_ip.strip(),
            "tanusha_ip": tanusha_ip.strip(),
        }
        save_config(new_cfg)
        st.success("✅ Saved! The app will now use these IPs automatically — no restart needed.")

    st.markdown("---")
    st.markdown("### Test Connections after Saving")
    if st.button("🔄 Test All Connections Now", width="stretch"):
        import requests as req
        saved_urls = get_current_urls()
        for name, url in saved_urls.items():
            try:
                r = req.get(url, timeout=5)
                st.success(f"**{name.capitalize()}** ({url}): ✅ Reachable")
            except req.exceptions.ConnectionError:
                st.error(f"**{name.capitalize()}** ({url}): ❌ Server not running")
            except req.exceptions.Timeout:
                st.error(f"**{name.capitalize()}** ({url}): ❌ Timed out")
            except Exception as e:
                if hasattr(e, 'response') and e.response is not None:
                    st.success(f"**{name.capitalize()}** ({url}): ✅ Reachable")
                else:
                    st.error(f"**{name.capitalize()}** ({url}): ❌ {type(e).__name__}")