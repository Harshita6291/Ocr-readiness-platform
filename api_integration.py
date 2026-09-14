"""
Team API Integration Layer
===========================
URLs are loaded dynamically from config.json (editable from Settings page).
No need to touch code when IPs change — just update from the app.

Mansi   → POST /<mansi_ip>:8000/analyze        (blur_score, contrast_score)
Krish   → POST /<krish_ip>:8002/scores         (matra_continuity_score, zone_integrity_score)
Vivek   → POST /<vivek_ip>:8001/analyze-image  (stroke_width, text_density)
Tanusha → POST /<tanusha_ip>:9001/analyze      (connected_component_stability_score, skew_penalty_score)
Yash    → Local                                (noise_score, resolution_score)
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from config_manager import get_urls

API_TIMEOUT = 10
FIELD_NAME  = "file"
KNOWN_ISSUES = []


# ── Helpers ────────────────────────────────────────────────────────────────
def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def _pil_to_bytes(img_pil: Image.Image) -> Tuple[bytes, str]:
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.read(), "image/png"

def _post_image(url: str, img_bgr: np.ndarray) -> Optional[Dict]:
    try:
        img_bytes, mime = _pil_to_bytes(_bgr_to_pil(img_bgr))
        files = {FIELD_NAME: ("image.png", img_bytes, mime)}
        resp  = requests.post(url, files=files, timeout=API_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

def _normalize(val) -> Optional[float]:
    try:
        return max(0.0, min(100.0, float(val)))
    except (TypeError, ValueError):
        return None

def _classify(score: float) -> str:
    if score >= 81: return "Excellent"
    if score >= 61: return "Good"
    if score >= 41: return "Average"
    return "Poor"


# ── Parsers ────────────────────────────────────────────────────────────────
def _parse_mansi(data):
    blur_result = contrast_result = None
    raw = _normalize(data.get("blur_score"))
    if raw is not None:
        blur_result = {"factor_name":"blur_score","score":round(raw,1),
            "status":_classify(raw),"description":f"Blur from Mansi API: {raw:.1f}",
            "source":"mansi_api","raw_value":data.get("blur_score"),"unit":""}
    raw = _normalize(data.get("contrast_score"))
    if raw is not None:
        contrast_result = {"factor_name":"contrast_score","score":round(raw,1),
            "status":_classify(raw),"description":f"Contrast from Mansi API: {raw:.1f}",
            "source":"mansi_api","raw_value":data.get("contrast_score"),"unit":""}
    return blur_result, contrast_result

def _parse_krish(data):
    matra_result = zone_result = None
    raw = _normalize(data.get("matra_continuity_score"))
    if raw is not None:
        matra_result = {"factor_name":"matra_continuity_score","score":round(raw,1),
            "status":_classify(raw),"description":f"Matra from Krish API: {raw:.1f}",
            "source":"krish_api","raw_value":data.get("matra_continuity_score"),"unit":""}
    raw = _normalize(data.get("zone_integrity_score"))
    if raw is not None:
        zone_result = {"factor_name":"zone_integrity_score","score":round(raw,1),
            "status":_classify(raw),"description":f"Zone from Krish API: {raw:.1f}",
            "source":"krish_api","raw_value":data.get("zone_integrity_score"),"unit":""}
    return matra_result, zone_result

def _parse_vivek(data):
    stroke_result = density_result = None
    raw = _normalize(data.get("stroke_width"))
    if raw is not None:
        stroke_result = {
            "factor_name": "stroke_width_score",
            "score": round(raw, 1),
            "status": _classify(raw),
            "description": f"Stroke from Vivek API: {raw:.1f}",
            "source": "vivek_api",
            "raw_value": data.get("stroke_raw_value", data.get("stroke_width")),
            "unit": data.get("stroke_unit", "ratio"),
            "details": data.get("stroke_details", {})
        }
    raw_td = _normalize(data.get("text_density"))
    if raw_td is not None:
        density_result = {
            "factor_name": "text_density_score",
            "score": round(raw_td, 1),
            "status": _classify(raw_td),
            "description": f"Density from Vivek API: {raw_td:.1f}",
            "source": "vivek_api",
            "raw_value": data.get("density_raw_value", data.get("text_density")),
            "unit": data.get("density_unit", "% text coverage"),
            "details": data.get("density_details", {})
        }
    return stroke_result, density_result

def _parse_tanusha(data):
    ccs_result = skew_result = None
    raw = _normalize(data.get("connected_component_stability_score"))
    if raw is not None:
        ccs_result = {"factor_name":"connected_component_stability_score","score":round(raw,1),
            "status":_classify(raw),"description":f"CC Stability from Tanusha API: {raw:.1f}",
            "source":"tanusha_api","raw_value":data.get("connected_component_stability_score"),"unit":""}
    raw = _normalize(data.get("skew_penalty_score"))
    if raw is not None:
        skew_result = {"factor_name":"skew_penalty_score","score":round(raw,1),
            "status":_classify(raw),"description":f"Skew from Tanusha API: {raw:.1f}",
            "source":"tanusha_api","raw_value":data.get("skew_penalty_score"),"unit":""}
    return ccs_result, skew_result


# ── Main orchestrator ──────────────────────────────────────────────────────
def call_all_team_apis(img_bgr, local_results):
    """
    Loads URLs fresh from config.json every time it is called.
    So IP changes take effect immediately without restarting the app.
    """
    urls   = get_urls()   # ← reads config.json fresh every call
    merged = dict(local_results)
    status = {}

    # Mansi
    data = _post_image(urls["mansi"], img_bgr)
    if data is not None:
        blur_r, contrast_r = _parse_mansi(data)
        if blur_r:     merged["blur_score"]     = blur_r;     status["blur_score"]     = "api ✅ Mansi"
        else:          status["blur_score"]      = "local ⚡ (Mansi returned unexpected format)"
        if contrast_r: merged["contrast_score"]  = contrast_r; status["contrast_score"] = "api ✅ Mansi"
        else:          status["contrast_score"]  = "local ⚡ (Mansi returned unexpected format)"
    else:
        status["blur_score"]     = "local ⚡ (Mansi API offline)"
        status["contrast_score"] = "local ⚡ (Mansi API offline)"

    # Krish
    data = _post_image(urls["krish"], img_bgr)
    if data is not None:
        matra_r, zone_r = _parse_krish(data)
        if matra_r: merged["matra_continuity_score"] = matra_r; status["matra_continuity_score"] = "api ✅ Krish"
        else:       status["matra_continuity_score"]  = "local ⚡ (Krish returned unexpected format)"
        if zone_r:  merged["zone_integrity_score"]    = zone_r;  status["zone_integrity_score"]   = "api ✅ Krish"
        else:       status["zone_integrity_score"]    = "local ⚡ (Krish returned unexpected format)"
    else:
        status["matra_continuity_score"] = "local ⚡ (Krish API offline)"
        status["zone_integrity_score"]   = "local ⚡ (Krish API offline)"

    # Vivek
    data = _post_image(urls["vivek"], img_bgr)
    if data is not None:
        stroke_r, density_r = _parse_vivek(data)
        if stroke_r:
            if (not stroke_r.get("details") or not stroke_r["details"].get("median_stroke_width_px")) and local_results.get("stroke_width_score", {}).get("details"):
                stroke_r["details"] = local_results["stroke_width_score"]["details"]
            if (stroke_r.get("raw_value") == stroke_r.get("score") or not stroke_r.get("raw_value")) and local_results.get("stroke_width_score", {}).get("raw_value") is not None:
                stroke_r["raw_value"] = local_results["stroke_width_score"]["raw_value"]
                stroke_r["unit"] = local_results["stroke_width_score"].get("unit", "ratio")
            merged["stroke_width_score"] = stroke_r
            status["stroke_width_score"] = "api ✅ Vivek"
        else:
            status["stroke_width_score"] = "local ⚡ (Vivek returned unexpected format)"
        if density_r:
            if not density_r.get("details") and local_results.get("text_density_score", {}).get("details"):
                density_r["details"] = local_results["text_density_score"]["details"]
            merged["text_density_score"]  = density_r
            status["text_density_score"] = "api ✅ Vivek"
        else:
            status["text_density_score"]  = "local ⚡ (Vivek returned unexpected format)"
    else:
        status["stroke_width_score"] = "local ⚡ (Vivek API offline)"
        status["text_density_score"] = "local ⚡ (Vivek API offline)"

    # Tanusha
    data = _post_image(urls["tanusha"], img_bgr)
    if data is not None:
        ccs_r, skew_r = _parse_tanusha(data)
        if ccs_r:  merged["connected_component_stability_score"] = ccs_r;  status["connected_component_stability_score"] = "api ✅ Tanusha"
        else:      status["connected_component_stability_score"]  = "local ⚡ (Tanusha returned unexpected format)"
        if skew_r: merged["skew_penalty_score"]                  = skew_r; status["skew_penalty_score"]                 = "api ✅ Tanusha"
        else:      status["skew_penalty_score"]                  = "local ⚡ (Tanusha returned unexpected format)"
    else:
        status["connected_component_stability_score"] = "local ⚡ (Tanusha API offline)"
        status["skew_penalty_score"]                  = "local ⚡ (Tanusha API offline)"

    # Yash — always local
    status["noise_score"]      = "local ✅ Yash"
    status["resolution_score"] = "local ✅ Yash"

    # Recompute OCR Readiness with final scores
    from factors import WEIGHTS, _clamp
    ocr_readiness = sum(merged[k]["score"] * w for k, w in WEIGHTS.items() if k in merged)
    merged["ocr_readiness_score"]  = round(_clamp(ocr_readiness), 1)
    merged["ocr_readiness_status"] = _classify(merged["ocr_readiness_score"])

    return merged, status


# Expose URLs for display in API Status page
def get_current_urls():
    return get_urls()
