"""
OCR Readiness Evaluation Platform
Factor computation engine — all 8 quality factors.

Each function accepts a numpy BGR image (OpenCV format) and returns
a dict with: score (0-100), status, description, details.

Team API integration notes:
  - Vivek  → stroke_width_score, text_density_score
  - Mansi  → blur_score, contrast_score
  - Krish  → matra_continuity_score, zone_integrity_score
  - Yash   → noise_score, resolution_score  (+ integration)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _classify(score: float) -> str:
    if score >= 81:
        return "Excellent"
    elif score >= 61:
        return "Good"
    elif score >= 41:
        return "Average"
    return "Poor"


def _clamp(v: float, lo=0.0, hi=100.0) -> float:
    return float(max(lo, min(hi, v)))


# ──────────────────────────────────────────────
# YASH — Noise Score (OCR Calibrated)
# ──────────────────────────────────────────────

def noise_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Estimates image noise using the high-frequency residual method,
    but ONLY measures it in flat/background regions — areas with no
    nearby text edges. This prevents dense, sharp text (which has
    naturally high edge-residual variance) from being misread as noise.

    Method:
      1. Detect edges (text strokes) via Canny.
      2. Dilate the edge mask to exclude a margin around every stroke.
      3. Compute Gaussian-blur residual, but only sample pixels in the
         remaining "flat" background area.
      4. std(residual in flat area) = true noise estimate.

    Score = clamp(100 − noise_std / 0.30, 0, 100)
    Calibration: clean scans have flat-region noise_std ~0-8;
    heavy scan/sensor noise pushes it to 15-30+.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {
            "factor_name": "noise_score",
            "score": 0,
            "status": "Poor",
            "description": "No image data available.",
            "raw_value": 0,
            "unit": "Noise level"
        }

    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img_bgr.astype(np.float32)

    # Exclude text-edge regions so they aren't counted as "noise"
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((7, 7), np.uint8))
    flat_mask = edges_dilated == 0

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray - blurred

    flat_residual = residual[flat_mask]
    if flat_residual.size < 50:
        # Fallback for images that are almost entirely text/edges
        noise_std = float(np.std(residual))
    else:
        noise_std = float(np.std(flat_residual))

    # Realistic continuous asymptotic curve:
    # Clean background (σ ~0-2) -> 93-96
    # Moderate noise (σ ~8-15) -> 60-75
    # Heavy noise (σ ~30-50) -> 18-30 (never flat 0.0)
    score = 16.0 + 80.0 * float(np.exp(-(noise_std / 16.0) ** 1.1))
    score = _clamp(score, lo=14.0, hi=96.0)

    return {
        "factor_name": "noise_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"Estimated background noise σ = {noise_std:.2f}. "
                       + ("Low noise — good OCR candidate." if score >= 70
                          else "Moderate noise detected." if score >= 45
                          else "High noise — apply denoising filter."),
        "raw_value": round(noise_std, 3),
        "unit": "σ (std of residual, flat regions only)",
    }


# ──────────────────────────────────────────────
# YASH — Resolution Score
# ──────────────────────────────────────────────

def resolution_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates image resolution for OCR.
    Uses total image pixels (megapixels) instead of only one dimension.
    Higher score = more character detail.
    """

    if img_bgr is None or img_bgr.size == 0:
        return {
            "factor_name": "resolution_score",
            "score": 0,
            "status": "Poor",
            "description": "No image data available.",
            "raw_value": 0,
            "unit": "Megapixels"
        }

    height, width = img_bgr.shape[:2]
    megapixels = (height * width) / 1_000_000

    # Continuous realistic saturation curve:
    # MP >= 2.0 -> 92-96 (never flat 100)
    # MP ~ 1.0 -> 84-88
    # MP ~ 0.35 -> 60-70
    # MP < 0.05 -> 20-35 (never flat 0.0 or 0.8)
    mp_pow = max(0.001, megapixels) ** 0.8
    k_pow = 0.55 ** 0.8
    score = 18.0 + 78.0 * (mp_pow / (mp_pow + k_pow))
    score = _clamp(score, lo=16.0, hi=96.5)

    return {
        "factor_name": "resolution_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description":
            f"Image size = {width} × {height} pixels "
            f"({megapixels:.2f} MP). " +
            (
                "High resolution with sufficient text details."
                if score >= 70 else
                "Moderate resolution. OCR may have minor issues."
                if score >= 35 else
                "Low resolution. Capture a higher-quality image."
            ),
        "raw_value": round(megapixels, 2),
        "unit": "Megapixels"
    }


# ──────────────────────────────────────────────
# MANSI — Blur Score (OCR calibrated)
# ──────────────────────────────────────────────
def blur_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Measure text sharpness using Laplacian variance.
    Higher score = sharper text.
    """

    if img_bgr is None or img_bgr.size == 0:
        return {
            "factor_name": "blur_score",
            "score": 0,
            "status": "Poor",
            "description": "No image data available.",
            "raw_value": 0,
            "unit": "Laplacian variance"
        }

    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    # Slight denoising to avoid noise creating fake edges
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Realistic sigmoidal sharpness curve:
    # Sharp document (lap_var > 300) -> 88-95 (never flat 100)
    # Moderate sharpness (lap_var ~ 80-150) -> 70-82
    # Defocus / blur (lap_var ~ 20-50) -> 45-60
    # Heavy blur (lap_var < 8) -> 16-28 (never flat 0.0)
    log_v = float(np.log1p(max(0.0, lap_var)))
    score = 15.0 + 81.0 / (1.0 + float(np.exp(-(log_v - 4.1) / 0.85)))
    score = _clamp(score, lo=14.0, hi=95.5) 

    return {
        "factor_name": "blur_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": (
            f"Laplacian variance = {lap_var:.2f}. " +
            (
                "Text edges are sharp and OCR readability is high."
                if score >= 70 else
                "Some blur is present. OCR may have minor errors."
                if score >= 35 else
                "Heavy blur detected. OCR accuracy may be poor."
            )
        ),
        "raw_value": round(lap_var, 2),
        "unit": "Laplacian variance"
    }


# ──────────────────────────────────────────────
# MANSI — Contrast Score (OCR calibrated)
# ──────────────────────────────────────────────
import cv2
import numpy as np
from typing import Dict, Any


def _classify(score: float) -> str:
    if score >= 81:
        return "Excellent"
    elif score >= 61:
        return "Good"
    elif score >= 41:
        return "Average"
    else:
        return "Poor"


def contrast_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    OCR-aware contrast score (Otsu foreground/background separation).
    Higher score = better OCR readability.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {
            "factor_name": "contrast_score",
            "score": 0,
            "status": "Poor",
            "description": "No image data available.",
            "raw_value": 0,
            "unit": "Intensity difference",
        }

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr.copy()
    h, w = gray.shape
    pixels = h * w

    # Otsu splits pixels into text (foreground) and page (background)
    # classes automatically, regardless of what fraction of the image
    # each class occupies — this is what fixes the sparse-text bug.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_pixels = gray[mask == 0]
    bg_pixels = gray[mask == 255]

    if fg_pixels.size == 0 or bg_pixels.size == 0:
        contrast = 0.0
    else:
        contrast = float(bg_pixels.mean() - fg_pixels.mean())

    # Realistic logistic contrast curve:
    # High contrast (contrast > 180) -> 92-96 (never flat 100)
    # Good separation (contrast ~ 130-160) -> 80-88
    # Average separation (contrast ~ 80-110) -> 55-70
    # Low contrast / faded (contrast < 40) -> 20-35 (never flat 0.0 or 5.0)
    score = 16.0 + 80.0 / (1.0 + float(np.exp(-(contrast - 95.0) / 28.0)))
    final_score = _clamp(score, lo=15.0, hi=96.0)

    return {
        "factor_name": "contrast_score",
        "score": round(final_score, 1),
        "status": _classify(final_score),
        "description": (
            f"Foreground/background separation = {contrast:.1f}. "
            + ("Excellent text/background separation." if final_score >= 81 else
               "Good contrast, suitable for OCR." if final_score >= 61 else
               "Average contrast; may need enhancement (e.g. CLAHE)." if final_score >= 41 else
               "Low contrast; OCR accuracy is likely to suffer.")
        ),
        "raw_value": round(contrast, 2),
        "unit": "Intensity difference",
    }

# ──────────────────────────────────────────────
# VIVEK — Stroke Width Score
# ──────────────────────────────────────────────

def stroke_width_score(img_bgr: np.ndarray) -> Dict[str, Any]:   
    """   
    Estimates stroke width relative to character height using:   
      1. Distance transform on Otsu binarized text (stroke radius via skeleton).   
      2. Connected components to estimate median character height (H_char).   
      3. Stroke-to-character-height ratio: R = W_stroke / H_char.   
       
    OCR Typographic Calibration:   
      - Ideal R is ~0.10 to 0.12 (stroke width ~11% of character height).   
      - R < 0.05: Stroke too thin / broken characters (under-inking/erosion).   
      - R > 0.20: Stroke too thick / loops fill in (over-inking/dilation).   
    """   
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)   
    h_img, w_img = gray.shape[:2]   
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)   
    valid_heights = []   
    for i in range(1, num_labels):   
        w = stats[i, cv2.CC_STAT_WIDTH]   
        h = stats[i, cv2.CC_STAT_HEIGHT]   
        area = stats[i, cv2.CC_STAT_AREA]   
   
        # Ignore outer page borders (box filling practically entire image)   
        if w >= 0.98 * w_img and h >= 0.98 * h_img:   
            continue   
        # Ignore tiny dust speckles   
        if h < 4 or w < 3 or area < 8:   
            continue   
        # Ignore extreme line separators / underlines   
        aspect = w / float(h)   
        if aspect > 25.0 or aspect < 0.04:   
            continue   
        valid_heights.append(h)   
   
    # Fallback for single large characters or tight crops where component occupies > 50%   
    if len(valid_heights) == 0:   
        for i in range(1, num_labels):   
            w = stats[i, cv2.CC_STAT_WIDTH]   
            h = stats[i, cv2.CC_STAT_HEIGHT]   
            if not (w >= 0.98 * w_img and h >= 0.98 * h_img) and h >= 4:   
                valid_heights.append(h)   
   
    # Ultimate fallback: bounding box of all non-zero binary pixels   
    if len(valid_heights) == 0:   
        pts = cv2.findNonZero(binary)   
        if pts is not None:   
            _, _, _, bh = cv2.boundingRect(pts)   
            if bh > 0:   
                valid_heights.append(bh)   
   
    # 3. Distance transform on text pixels for stroke radii   
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)   
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))   
    dilated = cv2.dilate(dist, kernel)   
    skeleton_mask = (dist == dilated) & (binary > 0)   
    stroke_radii = dist[skeleton_mask]   
   
    if len(stroke_radii) == 0 or len(valid_heights) == 0:   
        return {   
            "factor_name": "stroke_width_score",   
            "score": 50.0,   
            "status": "Average",   
            "description": "Could not detect distinct text strokes or character glyphs.",   
            "raw_value": 0.0,   
            "unit": "ratio",   
            "details": {   
                "median_stroke_width_px": 0.0,   
                "median_char_height_px": 0.0,   
                "stroke_to_height_ratio": 0.0   
            }   
        }   
   
    median_sw = float(np.median(stroke_radii)) * 2.0  # radius -> width   
    median_char_h = float(np.median(valid_heights))   
   
    # Guard against division by zero   
    ratio = median_sw / max(median_char_h, 1.0)

    # 4. Calibrated Score Calculation (Realistic typographic bounds ~20 to 95.5)
    if ratio <= 0.0:
        score = 45.0
    elif 0.08 <= ratio <= 0.15:
        score = 88.0 + 7.5 * (1.0 - abs(ratio - 0.115) / 0.035)
    elif ratio < 0.08:
        score = 22.0 + 66.0 * float((max(0.001, ratio) / 0.08) ** 1.3)
    else:
        score = 22.0 + 66.0 * float(np.exp(-0.5 * ((ratio - 0.15) / 0.11) ** 1.6))
    score = _clamp(score, lo=20.0, hi=95.5)

    # Description generator
    if ratio < 0.06:
        desc = f"Strokes are too thin ({median_sw:.1f} px, ratio {ratio:.2f}) — risk of broken character strokes."
    elif ratio > 0.22:
        desc = f"Strokes are heavy/bold ({median_sw:.1f} px, ratio {ratio:.2f}) — risk of character loops filling in."
    elif score >= 80:
        desc = f"Optimal stroke width ({median_sw:.1f} px, ~{ratio*100:.1f}% of char height {median_char_h:.1f} px)."
    else:
        desc = f"Acceptable stroke width ({median_sw:.1f} px, ratio {ratio:.2f})."

    return {   
        "factor_name": "stroke_width_score",   
        "score": round(score, 1),   
        "status": _classify(score),   
        "description": desc,   
        "raw_value": round(ratio, 3),   
        "unit": "stroke/height ratio",   
        "details": {   
            "median_stroke_width_px": round(median_sw, 2),   
            "median_char_height_px": round(median_char_h, 2),   
            "stroke_to_height_ratio": round(ratio, 3)   
        }   
    }   

# ──────────────────────────────────────────────
# VIVEK — Text Density Score
# ──────────────────────────────────────────────

def text_density_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Ratio of text pixels (dark foreground) to total image pixels.
    Uses Otsu thresholding.
    Ideal range for a text document: 5–35 % text pixel coverage.
    Score peaks at 20 % coverage and falls off towards 0 or 100 %.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_pixels = int(np.sum(binary > 0))
    total_pixels = binary.size
    density_pct = 100.0 * text_pixels / total_pixels
    # Realistic continuous curve peaked at 20% coverage (~22 to 95.0, never flat 100)
    score = 22.0 + 73.0 * float(np.exp(-0.5 * ((density_pct - 20.0) / 14.0) ** 2))
    score = _clamp(score, lo=22.0, hi=95.0)
    return {
        "factor_name": "text_density_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"Text coverage = {density_pct:.1f}% of image. "
                       + ("Optimal text coverage for OCR candidate." if score >= 80
                          else "Good text density, suitable for OCR." if score >= 60
                          else "Text is sparse or very dense." if score >= 40
                          else "Extremely sparse text coverage."),
        "raw_value": round(density_pct, 2),
        "unit": "% text pixel coverage",
    }


# ──────────────────────────────────────────────
# KRISH — Matra Continuity Score
# ──────────────────────────────────────────────

def matra_continuity_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Self-contained MCS using the same logic as the full pipeline.
    Fixed for handwriting and printed Hindi text.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    row_sums = np.sum(binary > 0, axis=1).astype(float)
    threshold_row = row_sums.max() * 0.05

    in_band = row_sums > threshold_row

    raw_bands = []
    start = None
    for i, val in enumerate(in_band):
        if val and start is None:
            start = i
        elif not val and start is not None:
            raw_bands.append((start, i))
            start = None
    if start is not None:
        raw_bands.append((start, h))

    # Merge bands close together to keep upper/lower matras united with the text line
    bands = []
    if raw_bands:
        merged = [raw_bands[0]]
        for b in raw_bands[1:]:
            last_start, last_end = merged[-1]
            cur_start, cur_end = b
            if cur_start - last_end <= 15:
                merged[-1] = (last_start, cur_end)
            else:
                merged.append(b)
        bands = [b for b in merged if (b[1] - b[0]) >= 12]

    if not bands:
        return {
            "factor_name": "matra_continuity_score",
            "score": 50.0,
            "status": "Average",
            "description": "No text bands detected.",
            "raw_value": 0,
            "unit": "MCS (0-100)",
        }

    band_scores = []
    for (r0, r1) in bands:
        band_h = r1 - r0
        if band_h < 8:
            continue

        band_patch = binary[r0:r1, :]
        search_top = int(band_h * 0.15)
        search_bot = max(search_top + 1, int(band_h * 0.45))
        row_proj = np.sum(band_patch[search_top:search_bot, :] > 0, axis=1)
        if len(row_proj) == 0:
            continue
        peak_y = search_top + int(np.argmax(row_proj))

        y1 = max(0, peak_y - 2)
        y2 = min(band_h, peak_y + 3)
        shiro_zone = band_patch[y1:y2, :]

        upper_zone   = band_patch[:search_top, :]
        lower_zone   = band_patch[max(search_bot, int(band_h * 0.70)):, :]
        middle_zone  = band_patch[search_bot:max(search_bot + 1, int(band_h * 0.70)), :]
        full_line    = band_patch

        # ── SCS: Shirorekha Continuity Score ────────────────────────
        if shiro_zone.size > 0:
            shiro_strip = (shiro_zone > 0).any(axis=0)
            body_strip = (middle_zone > 0).any(axis=0)
            word_mask = cv2.dilate(body_strip.astype(np.uint8)[None, :], np.ones((1, 5), np.uint8))[0] > 0
            word_cols = np.where(word_mask)[0]
            if len(word_cols) < 5:
                scs = 50.0
            else:
                shiro_in_words = shiro_strip[word_cols]
                coverage = float(shiro_in_words.mean())
                runs = []
                curr = 0
                for val in shiro_strip:
                    if val:
                        curr += 1
                    else:
                        if curr > 0:
                            runs.append(curr)
                        curr = 0
                if curr > 0:
                    runs.append(curr)
                mean_run = float(np.mean(runs)) if runs else 0.0
                run_factor = min(1.0, mean_run / 18.0)
                scs = float(np.clip(coverage * 60.0 * run_factor + min(40.0, mean_run * 1.5), 0, 100))
        else:
            scs = 50.0

        # ── MVS: Matra Visibility Score ──────────────────────────────
        def mvs(zone):
            if zone.size == 0:
                return 50.0
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                zone, connectivity=8)
            if n <= 1:
                return 100.0
            min_area = 4
            valid = sum(1 for i in range(1, n)
                        if stats[i, cv2.CC_STAT_AREA] >= min_area)
            total = n - 1
            if total == 0:
                return 100.0                                       
            return float(np.clip((valid / total) * 100, 0, 100))

        mvs_upper = mvs(upper_zone)
        mvs_lower = mvs(lower_zone)

        # ── RLR: Run-Length Regularity ───────────────────────────────
        runs = []
        for row in range(full_line.shape[0]):
            in_run = False
            length = 0
            for px in full_line[row]:
                if px > 0:
                    length += 1
                    in_run = True
                else:
                    if in_run and length >= 2:
                        runs.append(length)
                    length = 0
                    in_run = False
            if in_run and length >= 2:
                runs.append(length)

        if runs:
            runs_arr = np.array(runs, dtype=np.float32)
            mean_r = runs_arr.mean()
            if mean_r > 0:
                cov = runs_arr.std() / mean_r
                rlr = float(np.clip(100.0 * np.exp(-cov * 0.5), 0, 100))
            else:
                rlr = 0.0
        else:
            rlr = 0.0

        # ── BS: Baseline Stability ───────────────────────────────────
        if middle_zone.size > 0:
            ink = (middle_zone > 0).astype(np.float32)
            col_sum = ink.sum(axis=0) + 1e-6
            row_idx = np.arange(middle_zone.shape[0], dtype=np.float32)
            coms = (ink * row_idx[:, None]).sum(axis=0) / col_sum
            var = float(coms.var())
            bs = float(np.clip(100.0 * np.exp(-var / 30.0), 0, 100))
        else:
            bs = 50.0

        # ── MCS formula — Devanagari OCR weighted ────────────────────
        line_mcs = (0.65 * scs +
            0.15 * mvs_upper +
            0.10 * mvs_lower +
            0.05 * bs +
            0.05 * rlr)
        band_scores.append(float(np.clip(line_mcs, 0, 100)))

    if not band_scores:
        score = 50.0
    else:
        score = float(np.mean(band_scores))
        score = float(np.clip(score, 0, 100))

    status = ("Excellent" if score >= 81 else "Good" if score >= 61
              else "Average" if score >= 41 else "Poor")
    return {
        "factor_name": "matra_continuity_score",
        "score": round(score, 1),
        "status": status,
        "description": f"Shirorekha continuity and matra visibility = {score:.1f}/100. "
                       + ("Matra well-preserved." if score >= 70
                          else "Matra continuity acceptable." if score >= 45
                          else "Significant matra breaks — poor Devanagari OCR expected."),
        "raw_value": round(score, 2),
        "unit": "MCS (0-100)",
    }


# ──────────────────────────────────────────────
# KRISH — Zone Integrity Score (self-contained)
# ──────────────────────────────────────────────

def zone_integrity_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates the three-zone structural integrity of Devanagari text
    (Upper zone: matras/ascenders, Shirorekha: headline, Middle zone: character bodies,
    Lower zone: descenders/matras).
    Clean text has distinct, well-separated modifiers without heavy fragmentation or smear.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {
            "factor_name": "zone_integrity_score",
            "score": 50.0,
            "status": "Average",
            "description": "No image data available.",
            "raw_value": 0,
            "unit": "ZIS (0-100)",
        }

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    # Detect text lines
    row_sums = np.sum(binary > 0, axis=1).astype(float)
    if row_sums.max() == 0:
        return {
            "factor_name": "zone_integrity_score",
            "score": 50.0,
            "status": "Average",
            "description": "No ink detected in image.",
            "raw_value": 0,
            "unit": "ZIS (0-100)",
        }

    threshold_row = row_sums.max() * 0.05
    raw_bands = []
    start = None
    for i, val in enumerate(row_sums > threshold_row):
        if val and start is None:
            start = i
        elif not val and start is not None:
            raw_bands.append((start, i))
            start = None
    if start is not None:
        raw_bands.append((start, h))

    # Merge nearby bands within 15px to preserve unified line structure
    merged = [raw_bands[0]] if raw_bands else []
    for b in raw_bands[1:]:
        if b[0] - merged[-1][1] <= 15:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    bands = [b for b in merged if (b[1] - b[0]) >= 12]

    if not bands:
        return {
            "factor_name": "zone_integrity_score",
            "score": 50.0,
            "status": "Average",
            "description": "No text bands found for zone analysis.",
            "raw_value": 0,
            "unit": "ZIS (0-100)",
        }

    def score_zone(patch, is_modifier=False):
        if patch.size == 0 or patch.sum() == 0:
            return 80.0 if is_modifier else 40.0
        n, _, stats, _ = cv2.connectedComponentsWithStats(patch, connectivity=8)
        if n <= 1:
            return 80.0 if is_modifier else 40.0

        areas = stats[1:, cv2.CC_STAT_AREA]
        valid = np.sum(areas >= 8)
        noise = np.sum(areas < 6)
        total = len(areas)

        clean_ratio = valid / max(1, total)
        noise_ratio = noise / max(1, total)

        dist = cv2.distanceTransform(patch, cv2.DIST_L2, 3)
        nz = dist[dist > 0]
        cov = (nz.std() / nz.mean()) if len(nz) >= 5 and nz.mean() > 0 else 0.4

        score = 100.0 * clean_ratio - (noise_ratio * 40.0) - max(0.0, (cov - 0.4) * 50.0)
        return float(np.clip(score, 10.0, 100.0))

    line_scores = []
    for (r0, r1) in bands:
        bh = r1 - r0
        patch = binary[r0:r1, :]
        p_up  = patch[:max(1, int(bh * 0.22)), :]
        p_sh  = patch[int(bh * 0.22):int(bh * 0.36), :]
        p_mid = patch[int(bh * 0.36):int(bh * 0.72), :]
        p_low = patch[int(bh * 0.72):, :]

        z_up  = score_zone(p_up, is_modifier=True)
        z_sh  = score_zone(p_sh, is_modifier=False)
        z_mid = score_zone(p_mid, is_modifier=False)
        z_low = score_zone(p_low, is_modifier=True)

        lz = 0.25 * z_up + 0.30 * z_sh + 0.30 * z_mid + 0.15 * z_low
        line_scores.append(lz)

    score = float(np.mean(line_scores)) if line_scores else 50.0
    score = _clamp(score)
    status = _classify(score)

    return {
        "factor_name": "zone_integrity_score",
        "score": round(score, 1),
        "status": status,
        "description": f"Devanagari zone structural integrity = {score:.1f}/100. "
                       + ("All zones intact." if score >= 70
                          else "Some zone degradation." if score >= 45
                          else "Significant zone damage detected."),
        "raw_value": round(score, 2),
        "unit": "ZIS (0-100)",
    }


# ──────────────────────────────────────────────
# TANUSHA — Connected Component Stability Score
# ──────────────────────────────────────────────

def connected_component_stability_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Measures how uniform/stable the detected character (connected
    component) sizes are. Stable, well-segmented text has components
    of fairly consistent area; noisy or broken text produces a wide
    spread of component sizes.

    Method:
      1. Binarise (Otsu).
      2. Find connected components, drop very tiny noise specks
         (< 4 px area) and the background label.
      3. Compute coefficient of variation (CV = std/mean) of
         component areas.
      4. Score = clamp(100 − CV*40, 0, 100)
         Lower CV (more uniform components) → higher score.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    areas = areas[areas >= 4]  # drop tiny specks

    if len(areas) < 2:
        return {
            "factor_name": "connected_component_stability_score",
            "score": 50.0,
            "status": "Average",
            "description": "Not enough text components detected to evaluate stability.",
            "raw_value": len(areas),
            "unit": "components",
        }

    mean_area = float(np.mean(areas))
    std_area  = float(np.std(areas))
    cv_val    = std_area / mean_area if mean_area > 0 else 0

    score = _clamp(100.0 - max(0.0, cv_val - 0.35) * 45.0)
    return {
        "factor_name": "connected_component_stability_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"{len(areas)} components detected, size CV = {cv_val:.2f}. "
                       + ("Character sizes very consistent — clean segmentation." if score >= 70
                          else "Moderate variation in character sizes." if score >= 45
                          else "Highly inconsistent character sizes — likely noise or broken glyphs."),
        "raw_value": round(cv_val, 3),
        "unit": "coefficient of variation",
    }



# ──────────────────────────────────────────────
# TANUSHA — Skew Penalty Score
# ──────────────────────────────────────────────

def skew_penalty_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Estimates the skew (rotation) angle of text lines using Hough Line Transform
    on prominent text edges/baselines (and Shirorekha in Devanagari).
    If no lines are detected (e.g. single character or isolated glyph), skew is 0.0°.
    Score = clamp(100 − skew_deg * 6, 0, 100).
    """
    import math

    if img_bgr is None or img_bgr.size == 0:
        return {
            "factor_name": "skew_penalty_score",
            "score": 50.0,
            "status": "Average",
            "description": "No image data available to estimate skew.",
            "raw_value": 0,
            "unit": "° skew angle",
        }

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr.copy()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=25, maxLineGap=10)
    if lines is None or len(lines) == 0:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=5)

    skew_deg = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for l in lines:
            x1, y1, x2, y2 = l.flatten()[:4]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if dx == 0 and dy == 0:
                continue
            deg = math.degrees(math.atan2(dy, dx))
            # Text lines are predominantly horizontal: normalise to [-45, 45]
            if abs(deg) <= 45:
                angles.append(deg)
            elif deg > 45:
                angles.append(deg - 90)
            elif deg < -45:
                angles.append(deg + 90)
        if len(angles) >= 2:
            skew_deg = abs(float(np.median(angles)))
        elif len(angles) == 1:
            skew_deg = abs(float(angles[0]))

    if skew_deg > 45:
        skew_deg = 90 - skew_deg

    score = _clamp(100.0 - skew_deg * 6.0)
    return {
        "factor_name": "skew_penalty_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"Estimated skew = {skew_deg:.2f}°. "
                       + ("Image is well-aligned." if score >= 70
                          else "Slight tilt detected." if score >= 45
                          else "Significant skew — deskew before OCR."),
        "raw_value": round(skew_deg, 2),
        "unit": "° skew angle",
    }


# ──────────────────────────────────────────────
# Master scorer
# ──────────────────────────────────────────────

WEIGHTS = {
    "noise_score":                       0.12,
    "resolution_score":                  0.12,
    "blur_score":                        0.15,
    "contrast_score":                    0.12,
    "stroke_width_score":                0.08,
    "text_density_score":                0.08,
    "matra_continuity_score":            0.08,
    "zone_integrity_score":              0.05,
    "connected_component_stability_score": 0.10,
    "skew_penalty_score":                0.10,
}

FACTOR_FUNCTIONS = {
    "noise_score":            noise_score,
    "resolution_score":       resolution_score,
    "blur_score":             blur_score,
    "contrast_score":         contrast_score,
    "stroke_width_score":     stroke_width_score,
    "text_density_score":     text_density_score,
    "matra_continuity_score": matra_continuity_score,
    "zone_integrity_score":   zone_integrity_score,
    "connected_component_stability_score": connected_component_stability_score,
    "skew_penalty_score":     skew_penalty_score,
}

DISPLAY_NAMES = {
    "noise_score":            "Noise",
    "resolution_score":       "Resolution",
    "blur_score":             "Blur",
    "contrast_score":         "Contrast",
    "stroke_width_score":     "Stroke Width",
    "text_density_score":     "Text Density",
    "matra_continuity_score": "Matra Continuity",
    "zone_integrity_score":   "Zone Integrity",
    "connected_component_stability_score": "CC Stability",
    "skew_penalty_score":     "Skew Penalty",
}


def run_all_factors(img_bgr: np.ndarray) -> Dict[str, Any]:
    results = {}
    for key, fn in FACTOR_FUNCTIONS.items():
        try:
            results[key] = fn(img_bgr)
        except Exception as e:
            results[key] = {
                "factor_name": key,
                "score": 50.0,
                "status": "Error",
                "description": f"Computation error: {e}",
                "raw_value": None,
                "unit": "",
            }

    # OCR Readiness Score
    ocr_readiness = sum(
        results[k]["score"] * w for k, w in WEIGHTS.items()
    )
    ocr_readiness = round(_clamp(ocr_readiness), 1)
    results["ocr_readiness_score"] = ocr_readiness
    results["ocr_readiness_status"] = _classify(ocr_readiness)
    return results


# ──────────────────────────────────────────────
# Smarter Recommendations Engine
# ──────────────────────────────────────────────

RECOMMENDATIONS_CONFIG = {
    "noise_score": {
        "threshold": 50,
        "raw_name": "background noise σ",
        "units": "",
        "advice_ranges": [
            (20.0, "critically high", "The background noise is extremely severe. Hand-held capture artifacts or scanning dust detected. Try running a strong bilateral filter or scan the document again in better light."),
            (40.0, "moderately high", "Noise levels will probably corrupt punctuation marks and small text characters. We recommend running a median blur filter or Gaussian denoising."),
            (60.0, "noticeable", "Background noise is present. If OCR fails, consider applying a mild denoising filter."),
            (80.0, "mild", "Noise is low and mostly negligible for OCR. Minor optimization is possible but not critical."),
            (100.0, "mostly optimal", "The background is clean and noise is well within safe margins.")
        ],
        "positive": "background is clean and noise-free."
    },
    "resolution_score": {
        "threshold": 50,
        "raw_name": "dimensions",
        "units": " MP",
        "advice_ranges": [
            (20.0, "critically low", "The letter dimensions are too small in pixels (characters are pixelated). Capture the document at a minimum of 300 DPI or use a higher-resolution camera."),
            (40.0, "suboptimal", "Image details are lacking. OCR accuracy might drop on smaller footnotes. Upscale the image or scan closer."),
            (60.0, "acceptable but low", "The image size is barely sufficient. You could crop tighter around the text area to maximize pixel utilization."),
            (80.0, "good", "Resolution is sufficient for standard OCR. Minor details are well resolved."),
            (100.0, "mostly optimal", "Resolution is excellent, capturing character fine details cleanly.")
        ],
        "positive": "image provides generous details and megapixels for high character fidelity."
    },
    "blur_score": {
        "threshold": 50,
        "raw_name": "Laplacian variance",
        "units": "",
        "advice_ranges": [
            (20.0, "heavily blurred", "The document is completely out of focus or suffered from camera shake. Clean the lens and hold the camera steady, or use a scanner."),
            (40.0, "moderately blurred", "Text edges are soft and fuzzy, which will cause character confusion. Apply a sharpening/unsharp mask filter."),
            (60.0, "slightly blurred", "OCR should proceed but might fail on complex strokes or thin characters. Use a sharpening filter."),
            (80.0, "mildly soft", "Slight softness detected. Standard text will recognize fine, but fine prints might require sharpening."),
            (100.0, "mostly optimal", "Text edges are sharp and text transitions are well-defined.")
        ],
        "positive": "text edges are incredibly sharp and well-defined."
    },
    "contrast_score": {
        "threshold": 50,
        "raw_name": "intensity difference",
        "units": "",
        "advice_ranges": [
            (20.0, "critically low", "The text color is faded and nearly identical to the background. Apply local/adaptive binarization or manual thresholding."),
            (40.0, "low", "Text is hard to distinguish. Readability is compromised. Try increasing contrast or adjusting levels."),
            (60.0, "mediocre", "Contrast is acceptable but not optimal. Boost contrast settings slightly."),
            (80.0, "good", "Contrast is strong, text-background thresholding will separate characters well."),
            (100.0, "mostly optimal", "Contrast separation is pristine.")
        ],
        "positive": "text stands out boldly against the background."
    },
    "stroke_width_score": {
        "threshold": 50,
        "raw_name": "stroke-to-height ratio",
        "units": "",
        "advice_ranges": [
            (20.0, "heavily distorted", "Strokes are either extremely thin (< 5% char height) or merged (> 22% char height). OCR characters will bleed. Check the scan DPI setting or binarization threshold."),
            (40.0, "suboptimal", "Text strokes are outside the ideal 10-12% character height range. Adjust the print size or scale of the capture."),
            (60.0, "marginal", "Strokes are marginally thin or thick relative to character height. OCR might suffer minor errors."),
            (80.0, "good", "Strokes are within optimal ratio limits for clean character contours."),
            (100.0, "mostly optimal", "Strokes match standard typographic proportions (~10-12% of character height).")
        ],
        "positive": "character strokes are perfectly proportioned relative to character height."
    },
    "text_density_score": {
        "threshold": 40,
        "raw_name": "text coverage",
        "units": "%",
        "advice_ranges": [
            (20.0, "highly unusual", "The text coverage indicates the image is mostly blank margins or overly packed. Crop tightly around the content block to exclude blank spaces."),
            (40.0, "imbalanced", "Text layout density is slightly off. Check if the region has large blank backgrounds or borders."),
            (60.0, "marginal", "Slight imbalance in text distribution. Consider cropping out empty outer borders."),
            (80.0, "good", "Balanced text distribution across the document surface."),
            (100.0, "mostly optimal", "Optimal text to background ratio.")
        ],
        "positive": "text layout and spacing are perfectly balanced."
    },
    "matra_continuity_score": {
        "threshold": 50,
        "raw_name": "MCS score",
        "units": "",
        "advice_ranges": [
            (20.0, "critically fragmented", "Severe fragmentation of Devanagari shirorekha (headline). Tesseract will missegment words. Ensure the document is clean, flat, and free of creases."),
            (40.0, "compromised", "Shirorekha lines have frequent breaks. The text might be hand-written/cursive or scanned poorly. Apply morphological closing to bridge gaps."),
            (60.0, "minor gaps", "Some matra breaks are present. A clean binarized version of the document will improve alignment."),
            (80.0, "good", "Headline alignment is mostly steady with minor gaps."),
            (100.0, "mostly optimal", "Shirorekha line exhibits near perfect continuity.")
        ],
        "positive": "headlines and upper Devanagari lines are continuous and intact."
    },
    "zone_integrity_score": {
        "threshold": 50,
        "raw_name": "ZIS score",
        "units": "",
        "advice_ranges": [
            (20.0, "critically degraded", "The Devanagari three-zone structure is broken. Modifiers (matras) are clipped or merged with the character bodies. Avoid clipping text boundaries."),
            (40.0, "poorly preserved", "Modifiers are poorly separated from characters. OCR mapping of vowels will fail. Check resolution and thresholding."),
            (60.0, "fair", "Modifiers are mostly separated, but zone transitions are noisy. Use adaptive thresholding to clean zone boundaries."),
            (80.0, "good", "Zone separation is clear. Upper and lower modifiers are easily locatable."),
            (100.0, "mostly optimal", "Zone partitioning is exceptionally clean.")
        ],
        "positive": "upper, middle, and lower text modifier zones are structurally intact and distinct."
    },
    "connected_component_stability_score": {
        "threshold": 50,
        "raw_name": "size CV",
        "units": "",
        "advice_ranges": [
            (20.0, "critically unstable", "Extremely high variance in character component sizes. Likely caused by heavy pepper/speckle noise or fragmented words. Clean small components."),
            (40.0, "unstable", "Inconsistent glyph sizes. Character segmentation will fail. Check for spelling/font variations or scanner artifacts."),
            (60.0, "moderately uniform", "Some size variance is present. Character structure is mostly readable."),
            (80.0, "good", "Glyph sizes are reasonably uniform across the document."),
            (100.0, "mostly optimal", "Component size distribution is exceptionally stable.")
        ],
        "positive": "character sizes are highly uniform, providing clean character boundaries."
    },
    "skew_penalty_score": {
        "threshold": 50,
        "raw_name": "skew angle",
        "units": "°",
        "advice_ranges": [
            (20.0, "severely skewed", "The document has a large rotation. Tesseract cannot segment lines correctly. Rotate/deskew the image before running OCR."),
            (40.0, "moderately tilted", "Some line overlapping may cause OCR line skips. Deskew the image."),
            (60.0, "slightly tilted", "A slight angle is present. Deskewing is recommended for maximum accuracy."),
            (80.0, "well-aligned", "Very minor tilt. Safe for standard OCR engines."),
            (100.0, "mostly optimal", "Text lines are perfectly level.")
        ],
        "positive": "document orientation is perfectly aligned."
    }
}


def generate_recommendations(results: Dict[str, Any]) -> list:
    recs = []
    
    # ── 1. Cross-factor Rules ──
    blur_score = results.get("blur_score", {}).get("score", 100.0)
    noise_score = results.get("noise_score", {}).get("score", 100.0)
    res_score = results.get("resolution_score", {}).get("score", 100.0)
    matra_score = results.get("matra_continuity_score", {}).get("score", 100.0)
    zone_score = results.get("zone_integrity_score", {}).get("score", 100.0)

    blur_raw = results.get("blur_score", {}).get("raw_value", 0.0)
    noise_raw = results.get("noise_score", {}).get("raw_value", 0.0)
    res_raw = results.get("resolution_score", {}).get("raw_value", 0.0)
    matra_raw = results.get("matra_continuity_score", {}).get("raw_value", 0.0)
    zone_raw = results.get("zone_integrity_score", {}).get("raw_value", 0.0)

    # Blur + Noise low
    if blur_score < 45 and noise_score < 45:
        recs.append(
            f"⚠️ **Consider rescanning/recapturing the document entirely** — Multiple fundamental quality issues: "
            f"both blur (unstable edges, Laplacian var = {blur_raw}) and background noise (σ = {noise_raw}) are at critical levels."
        )
    # Resolution low but others fine
    elif res_score < 50 and blur_score >= 60 and noise_score >= 60:
        recs.append(
            f"💡 **Crop tighter around the text area** — The resolution score is low ({res_score}/100, MP = {res_raw}), "
            f"but the image is sharp and quiet. Cropping out blank margins will maximize the pixel coverage of the characters."
        )
    
    # Hindi structure degradation
    if matra_score < 45 and zone_score < 45:
        recs.append(
            f"⚠️ **Severe Devanagari structure distortion** — Both Matra continuity ({matra_score}/100) and "
            f"zone integrity ({zone_score}/100) are low. OCR segmentations will likely break. "
            f"A higher resolution scan or preprocessing (like binarization) is strongly recommended."
        )

    # ── 2. Identify Top 3 Weakest Factors ──
    # Sort factors present in results by their score ascending
    factor_scores = []
    for key in RECOMMENDATIONS_CONFIG.keys():
        if key in results:
            factor_scores.append((key, results[key]["score"]))
            
    # Sort ascending: lowest score first
    factor_scores.sort(key=lambda x: x[1])
    weakest_keys = [x[0] for x in factor_scores[:3]]

    # ── 3. Identify Target Keys for Recommendations ──
    # Keys below threshold or inside the weakest 3
    target_keys = set()
    for key, cfg in RECOMMENDATIONS_CONFIG.items():
        if key in results:
            if results[key]["score"] < cfg["threshold"] or key in weakest_keys:
                target_keys.add(key)

    # Sort target keys by priority descending: (threshold - score) * weight
    prioritized_keys = list(target_keys)
    prioritized_keys.sort(
        key=lambda k: (RECOMMENDATIONS_CONFIG[k]["threshold"] - results[k]["score"]) * WEIGHTS[k],
        reverse=True
    )

    # ── 4. Generate Warning/Tip Recommendations (only if score < 81) ──
    warning_recs = []
    
    # Helper to find the matched range adjective and advice
    def get_range_info(key, score):
        cfg = RECOMMENDATIONS_CONFIG[key]
        for max_val, adj, adv in cfg["advice_ranges"]:
            if score <= max_val:
                return adj, adv
        return cfg["advice_ranges"][-1][1], cfg["advice_ranges"][-1][2]

    for key in prioritized_keys:
        score = results[key]["score"]
        if score >= 81.0:
            continue
            
        cfg = RECOMMENDATIONS_CONFIG[key]
        display = DISPLAY_NAMES.get(key, key)
        raw_val = results[key].get("raw_value", 0.0)
        adj, advice = get_range_info(key, score)
        
        warn_str = (
            f"🔧 **Your {display} score is {score} ({cfg['raw_name']} = {raw_val}{cfg['units']})** — "
            f"this is {adj}. {advice}"
        )
        warning_recs.append(warn_str)

    # ── 5. Generate Positive Acknowledgements ──
    positive_recs = []
    for key, cfg in RECOMMENDATIONS_CONFIG.items():
        if key in results:
            score = results[key]["score"]
            display = DISPLAY_NAMES.get(key, key)
            raw_val = results[key].get("raw_value", 0.0)
            
            if score >= 81.0:
                if key in weakest_keys:
                    # Excellent, but still one of the top-weakest! Include a constructive tip.
                    adj, advice = get_range_info(key, score)
                    pos_str = (
                        f"✅ **Excellent {display}** ({score}/100) — no action needed. "
                        f"({display} is among your lowest scores; {cfg['raw_name']} is {raw_val}{cfg['units']}. Tip to optimize further: {advice})"
                    )
                else:
                    pos_str = f"✅ **Excellent {display}** ({score}/100) — no action needed. {cfg['positive']}"
                positive_recs.append((pos_str, score))

    # Sort positives by score descending (highest-scoring positive first)
    positive_recs.sort(key=lambda x: x[1], reverse=True)
    positive_recs_str = [x[0] for x in positive_recs]

    # Combine all recommendations
    recs.extend(warning_recs)
    recs.extend(positive_recs_str)
    
    if not recs:
        recs.append("✅ All factors are within acceptable ranges. Image looks ready for OCR!")
        
    return recs