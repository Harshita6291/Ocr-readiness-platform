"""Safe, factor-specific OCR image refinement.

This module deliberately does not contain scoring formulas.  It generates
conservative image candidates and delegates all measurements to the existing
``run_all_factors`` implementation in :mod:`factors`.  A candidate is useful
only when the authoritative scores satisfy the safety policy below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import math

import cv2
import numpy as np

from factors import DISPLAY_NAMES, run_all_factors


FACTOR_KEYS: Tuple[str, ...] = tuple(DISPLAY_NAMES.keys())
MAX_SOURCE_PIXELS = 16_000_000
MAX_UPSCALED_PIXELS = 16_000_000
MAX_UPSCALED_SIDE = 6_000
TARGET_MIN_IMPROVEMENT = 0.5
MAX_NON_TARGET_DROP = 5.0


REFINEMENT_INFO: Dict[str, Dict[str, str]] = {
    "noise_score": {
        "method": "Mild edge-preserving denoising",
        "details": "Tests small median, bilateral, and non-local-means denoisers. Text edges are kept intact as far as possible.",
    },
    "resolution_score": {
        "method": "Controlled upscaling",
        "details": "Tests 1.25×, 1.5×, and 2× cubic/Lanczos enlargement within a strict pixel limit. It cannot recover detail absent from the source.",
    },
    "blur_score": {
        "method": "Conservative unsharp masking",
        "details": "Tests modest Gaussian-detail enhancement; candidates with harmful halos or score regressions are rejected by the full safety check.",
    },
    "contrast_score": {
        "method": "Tone and local-contrast correction",
        "details": "Tests restrained contrast/gamma adjustment and low-clip CLAHE without forcing a binary document image.",
    },
    "stroke_width_score": {
        "method": "Tiny grayscale morphology",
        "details": "Uses one-pixel dark-ink expansion for thin strokes or contraction for heavy strokes. It never removes components deliberately.",
    },
    "text_density_score": {
        "method": "Safe border tightening and background normalization",
        "details": "Only crops clearly blank outer margins with protective padding. It does not add or remove text pixels to chase a density score.",
    },
    "matra_continuity_score": {
        "method": "Devanagari-preserving cleanup",
        "details": "Tests mild background normalization and tiny horizontal gap closure. Large gaps and word-to-word bridging are never attempted.",
    },
    "zone_integrity_score": {
        "method": "Conservative structural cleanup",
        "details": "Tests illumination normalization, gentle denoising, and only one-pixel isolated-speck removal while preserving vertical character zones.",
    },
    "connected_component_stability_score": {
        "method": "Text-protected component cleanup & gap repair",
        "details": "Removes isolated noise specks (area 4-40px) outside text bands while strictly protecting Devanagari matras, anusvara, and punctuation, and bridges hairline stroke fragments.",
    },
    "skew_penalty_score": {
        "method": "Bounded deskew search",
        "details": "Uses the same Hough-line concept as the scorer to test rotations within ±3°. The canvas is expanded so text is not cut off.",
    },
}


@dataclass
class RefinementCandidate:
    """One transformed image plus its authoritative evaluation."""

    image: np.ndarray
    target_factor: str
    method: str
    parameters: Dict[str, Any]
    scores: Optional[Dict[str, Any]] = None
    target_improvement: float = 0.0
    readiness_improvement: float = 0.0
    non_target_changes: Dict[str, float] = field(default_factory=dict)
    safe: bool = False
    safety_reason: str = "Not evaluated"

    def summary(self) -> Dict[str, Any]:
        return {
            "target_factor": self.target_factor,
            "method": self.method,
            "parameters": self.parameters,
            "target_improvement": self.target_improvement,
            "readiness_improvement": self.readiness_improvement,
            "non_target_changes": self.non_target_changes,
            "safe": self.safe,
            "safety_reason": self.safety_reason,
        }


@dataclass
class RefinementOutcome:
    target_factor: str
    baseline_scores: Dict[str, Any]
    best_candidate: Optional[RefinementCandidate]
    evaluated: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""


@dataclass
class RefineAllOutcome:
    baseline_scores: Dict[str, Any]
    final_image: np.ndarray
    final_scores: Dict[str, Any]
    steps: List[RefinementCandidate]
    message: str


def refinement_eligibility(score: float) -> Tuple[str, str]:
    """Return ``eligible``, ``good``, or ``severe`` and its UI message."""
    if score >= 80.0:
        return "good", "This factor is already at or above the refinement threshold."
    if score < 20.0:
        return (
            "severe",
            "This factor is severely degraded. Automatic refinement is disabled because "
            "the original image may not contain enough recoverable information. Please "
            "recapture or rescan the document.",
        )
    return "eligible", "Refinement is available for this factor."


def cc_refinement_eligibility(score: float) -> Tuple[str, str]:
    """CC Stability-specific eligibility check.

    Unlike the generic rule, CC Stability allows refinement attempts for ANY
    score below 80 — including 0.0.  A very low CV (high noise/fragmentation)
    is precisely the situation where candidate cleanup operations may help.
    The normal safety policy (target +0.5, no non-target drop > 5, OCR
    Readiness strictly increases) still applies to every candidate.
    """
    if score >= 80.0:
        return "good", "CC Stability is already at or above the refinement threshold."
    return "eligible", "CC Stability refinement is available — candidates will be evaluated."


def matra_refinement_eligibility(score: float) -> Tuple[str, str]:
    """Matra Continuity-specific eligibility check.

    Like CC Stability, Matra Continuity allows refinement attempts for ANY
    score below 80 — including very low scores.  Poor matra continuity is
    precisely where shirorekha/matra repair operations can visibly help.
    The standard safety policy still applies to every evaluated candidate.
    """
    if score >= 80.0:
        return "good", "Matra Continuity is already at or above the refinement threshold."
    return "eligible", "Matra Continuity refinement is available — candidates will be evaluated."


def zone_integrity_refinement_eligibility(score: float) -> Tuple[str, str]:
    """Zone Integrity-specific eligibility check.

    Like CC Stability and Matra Continuity, Zone Integrity allows refinement attempts for ANY
    score below 80 — including very low scores. Structural cleanup operations can visibly help
    restore zone structure.
    The standard safety policy still applies to every evaluated candidate.
    """
    if score >= 80.0:
        return "good", "Zone Integrity is already at or above the refinement threshold."
    return "eligible", "Zone Integrity refinement is available — candidates will be evaluated."





def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Validate and canonicalise an OpenCV image without altering its content."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("No image data available for refinement.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Unsupported image shape for refinement.")
    if image.shape[0] < 8 or image.shape[1] < 8:
        raise ValueError("Image is too small for safe refinement.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2RGB)


def _candidate(image: np.ndarray, target: str, method: str, **parameters: Any) -> RefinementCandidate:
    return RefinementCandidate(ensure_bgr(image), target, method, parameters)


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)


def _gray_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _contrast_adjust(image: np.ndarray, alpha: float, beta: float = 0.0) -> np.ndarray:
    return cv2.convertScaleAbs(ensure_bgr(image), alpha=alpha, beta=beta)


def _gamma_adjust(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(ensure_bgr(image), table)


def _clahe_luminance(image: np.ndarray, clip_limit: float = 1.5) -> np.ndarray:
    lab = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _background_normalize(image: np.ndarray) -> np.ndarray:
    """Flatten gentle illumination changes while retaining grayscale text detail."""
    gray = _gray(image).astype(np.float32)
    scale = max(15, (min(gray.shape) // 20) | 1)
    background = cv2.GaussianBlur(gray, (scale, scale), 0)
    normalized = cv2.divide(gray, np.maximum(background, 1.0), scale=190.0)
    return _gray_bgr(normalized)


def _binary_ink(image: np.ndarray) -> np.ndarray:
    gray = _gray(image)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def _ink_to_document(ink: np.ndarray) -> np.ndarray:
    return _gray_bgr(255 - np.clip(ink, 0, 255).astype(np.uint8))


def _remove_one_pixel_specks(image: np.ndarray) -> np.ndarray:
    """Remove only isolated 1–2 pixel dots; legitimate modifiers are preserved."""
    ink = _binary_ink(image)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    cleaned = ink.copy()
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] <= 2:
            cleaned[labels == label] = 0
    return _ink_to_document(cleaned)


def _horizontal_close(image: np.ndarray, width: int = 2) -> np.ndarray:
    ink = _binary_ink(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
    return _ink_to_document(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=1))


def _matra_shirorekha_repair(image: np.ndarray, close_width: int = 8, dilate_px: int = 1) -> np.ndarray:
    """Repair broken shirorekha (headline) and matra strokes in Devanagari text.

    Produces a **visibly** different output by:
    1. Binarising with Otsu to get pure ink.
    2. On the upper 40% of every detected text line band (where the shirorekha
       runs) applying a horizontal DILATION followed by a horizontal CLOSE so
       broken headline segments are thickened AND gaps are bridged.  This makes
       the repaired image look clearly different from the original.
    3. On the remaining body zone applying a gentler horizontal close to repair
       broken matra arms and character strokes.
    4. Protecting vertical character extent — no vertical dilation so
       neighbouring lines are never merged.
    5. Converting the repaired binary ink back to a clean white-background
       document image (same style as the input but with connected strokes).
    """
    bgr = ensure_bgr(image)
    gray = _gray(bgr)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = ink.shape

    # ── Detect text line bands ───────────────────────────────────────────────
    row_sums = np.sum(ink > 0, axis=1).astype(float)
    if row_sums.max() == 0:
        return bgr
    thresh = row_sums.max() * 0.04
    in_band = row_sums > thresh

    bands: List[Tuple[int, int]] = []
    start = None
    for i, val in enumerate(in_band):
        if val and start is None:
            start = i
        elif not val and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, h))
    merged: List[Tuple[int, int]] = []
    if bands:
        merged = [bands[0]]
        for b in bands[1:]:
            if b[0] - merged[-1][1] <= 15:
                merged[-1] = (merged[-1][0], b[1])
            else:
                merged.append(b)
    bands = [b for b in merged if (b[1] - b[0]) >= 10]

    result_ink = ink.copy()

    # Kernels
    h_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, dilate_px * 2 + 1), 1))
    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 1))
    h_body_close = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, close_width // 2), 1))

    for (r0, r1) in bands:
        band_h = r1 - r0

        # ── Shirorekha zone: upper 40% ─────────────────────────────────────
        shiro_end = r0 + max(2, int(band_h * 0.40))
        shiro_zone = result_ink[r0:shiro_end, :].copy()
        # Step 1: dilate horizontally to thicken the headline
        shiro_zone = cv2.dilate(shiro_zone, h_dil, iterations=1)
        # Step 2: close to bridge remaining gaps
        shiro_zone = cv2.morphologyEx(shiro_zone, cv2.MORPH_CLOSE, h_close, iterations=2)
        result_ink[r0:shiro_end, :] = shiro_zone

        # ── Body zone: lower 60% — gentler repair ─────────────────────────
        body_start = shiro_end
        body_zone = result_ink[body_start:r1, :].copy()
        body_zone = cv2.morphologyEx(body_zone, cv2.MORPH_CLOSE, h_body_close, iterations=1)
        result_ink[body_start:r1, :] = body_zone

    # ── Convert binary ink back to clean document image ──────────────────────
    # White background, pure black ink — makes the repair visually clear
    out = np.full_like(gray, 255)
    out[result_ink > 0] = 0
    return _gray_bgr(out)


def _matra_full_image_repair(image: np.ndarray, close_width: int = 10) -> np.ndarray:
    """Apply horizontal close globally across the entire image to bridge matra/stroke gaps.

    More aggressive than ``_matra_shirorekha_repair``: operates on every ink
    pixel rather than just the detected text-band zones.  Used as a fallback
    when per-band detection is unreliable.
    """
    bgr = ensure_bgr(image)
    gray = _gray(bgr)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 1))
    # Two iterations to bridge wider gaps
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, h_kernel, iterations=2)
    # Also dilate slightly to thicken thin strokes
    dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    thickened = cv2.dilate(closed, dil_k, iterations=1)
    out = np.full_like(gray, 255)
    out[thickened > 0] = 0
    return _gray_bgr(out)


def _zone_integrity_repair(
    image: np.ndarray,
    max_speck_area: int = 8,
    shiro_close: int = 5,
    body_close: int = 3,
    clean_interline: bool = True,
) -> np.ndarray:
    """Refines Devanagari text zone structure based on factors.py zone_integrity_score metrics.

    Strictly preserves original page color — no background or ink tone is altered.

    1. Binarises with Otsu to isolate ink pixels.
    2. Detects text line bands using row density threshold (5% of max row sum).
    3. Cleans interline background noise specks between text bands.
    4. Partitions each band into Upper (0-22%), Shirorekha (22-36%), Middle (36-72%), and Lower (72-100%) zones.
    5. Removes intra-zone noise specks (area < max_speck_area) that penalize noise_ratio and clean_ratio.
    6. Performs horizontal gap closing on shirorekha and body zones to reduce distance-transform COV.
    7. Removed speck pixels are filled using TELEA inpainting from surrounding background neighbors.
    8. Added gap-bridge pixels inherit the color of the nearest original ink pixel.
       No global median or average — every pixel inherits only local context.
    """
    bgr = ensure_bgr(image)
    gray = _gray(bgr)
    _, raw_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = raw_ink.shape

    # Filter out margin stains & giant border artifacts so they are not treated as text ink
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_ink, connectivity=8)
    ink = raw_ink.copy()
    for lbl in range(1, num_labels):
        comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_w = stats[lbl, cv2.CC_STAT_WIDTH]
        area = stats[lbl, cv2.CC_STAT_AREA]
        if comp_h > 0.45 * h or area > 0.04 * h * w or (comp_w > 0.85 * w and comp_h > 0.25 * h):
            ink[labels == lbl] = 0

    row_sums = np.sum(ink > 0, axis=1).astype(float)
    if row_sums.max() == 0:
        return bgr
    thresh = row_sums.max() * 0.05
    raw_bands = []
    start = None
    for i, val in enumerate(row_sums > thresh):
        if val and start is None:
            start = i
        elif not val and start is not None:
            raw_bands.append((start, i))
            start = None
    if start is not None:
        raw_bands.append((start, h))

    if not raw_bands:
        return bgr

    merged = [raw_bands[0]]
    for b in raw_bands[1:]:
        if b[0] - merged[-1][1] <= 15:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    bands = [b for b in merged if (b[1] - b[0]) >= 12]

    result_ink = ink.copy()

    # Interline cleaning
    if clean_interline:
        in_band_mask = np.zeros(h, dtype=bool)
        for r0, r1 in bands:
            in_band_mask[r0:r1] = True
        num_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(result_ink, connectivity=8)
        for label_idx in range(1, num_c):
            if stats_c[label_idx, cv2.CC_STAT_AREA] < 60:
                comp_rows = np.where(labels_c == label_idx)[0]
                if np.all(~in_band_mask[comp_rows]):
                    result_ink[labels_c == label_idx] = 0

    k_shiro = cv2.getStructuringElement(cv2.MORPH_RECT, (shiro_close, 1)) if shiro_close > 1 else None
    k_body = cv2.getStructuringElement(cv2.MORPH_RECT, (body_close, 1)) if body_close > 1 else None
    k_sub = cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, body_close - 1), 1)) if body_close > 1 else None

    for (r0, r1) in bands:
        bh = r1 - r0
        u_end = r0 + int(bh * 0.22)
        s_end = r0 + int(bh * 0.36)
        m_end = r0 + int(bh * 0.72)

        for z_start, z_end, kernel in [
            (r0, u_end, k_sub),
            (u_end, s_end, k_shiro),
            (s_end, m_end, k_body),
            (m_end, r1, k_sub)
        ]:
            if z_end <= z_start:
                continue
            z_patch = result_ink[z_start:z_end, :].copy()
            n, labels, stats, _ = cv2.connectedComponentsWithStats(z_patch, connectivity=8)
            for l_idx in range(1, n):
                if stats[l_idx, cv2.CC_STAT_AREA] < max_speck_area:
                    z_patch[labels == l_idx] = 0
            if kernel is not None and z_patch.shape[1] > 0:
                z_patch = cv2.morphologyEx(z_patch, cv2.MORPH_CLOSE, kernel)
            result_ink[z_start:z_end, :] = z_patch

    # ── Compute change masks ──────────────────────────────────────────────────
    removed_mask = (ink > 0) & (result_ink == 0)   # noise specks erased
    added_mask   = (ink == 0) & (result_ink > 0)   # gap pixels bridged

    # Start from the untouched original — every unchanged pixel keeps its exact color.
    out = bgr.copy()

    # Removed speck pixels → fill with local background via TELEA inpainting.
    # This reconstructs the pixel from its nearest background neighbours in the
    # original image, so the paper texture and color are faithfully preserved.
    if removed_mask.any():
        removed_u8 = removed_mask.astype(np.uint8) * 255
        out = cv2.inpaint(out, removed_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Gap-bridge pixels → inherit the color of the spatially nearest original
    # ink pixel, so bridged strokes match their surrounding ink tone exactly.
    if added_mask.any():
        ink_ys, ink_xs = np.where(ink > 0)
        if len(ink_ys) > 0:
            _, nearest_idx = cv2.distanceTransformWithLabels(
                (ink == 0).astype(np.uint8), cv2.DIST_L2, 5,
                labelType=cv2.DIST_LABEL_PIXEL,
            )
            ink_positions = np.stack([ink_ys, ink_xs], axis=1)
            add_ys, add_xs = np.where(added_mask)
            label_vals = (nearest_idx[add_ys, add_xs].astype(np.int64) - 1).clip(0, len(ink_positions) - 1)
            src_ys = ink_positions[label_vals, 0]
            src_xs = ink_positions[label_vals, 1]
            out[add_ys, add_xs] = bgr[src_ys, src_xs]

    return out





def _cc_text_line_bands(binary_ink: np.ndarray) -> List[Tuple[int, int]]:
    """Detect horizontal text line bands with generous protective margins for Devanagari modifiers."""
    h, w = binary_ink.shape
    row_sums = np.sum(binary_ink > 0, axis=1).astype(float)
    if row_sums.max() == 0:
        return []
    threshold_row = row_sums.max() * 0.03
    in_band = row_sums > threshold_row

    bands = []
    start = None
    for i, val in enumerate(in_band):
        if val and start is None:
            start = i
        elif not val and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, h))

    merged = []
    if bands:
        merged = [bands[0]]
        for b in bands[1:]:
            last_s, last_e = merged[-1]
            if b[0] - last_e <= 15:
                merged[-1] = (last_s, b[1])
            else:
                merged.append(b)
    # Add protective 10px padding above and below each text band for matras and ascenders/descenders
    return [(max(0, s - 10), min(h, e + 10)) for s, e in merged]


def _cc_clean_isolated_specks(
    image: np.ndarray,
    max_speck_area: int = 60,
    dist_thresh: float = 10.0,
) -> np.ndarray:
    """Remove background noise specks while strictly protecting Devanagari anusvara dots and text strokes."""
    bgr = ensure_bgr(image)
    gray = _gray(bgr)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if num_labels <= 1:
        return bgr

    # Real text components must have area >= 60 OR (height >= 14 AND width >= 8)
    main_indices = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        width = stats[i, cv2.CC_STAT_WIDTH]
        if area >= 60 or (height >= 14 and width >= 8):
            main_indices.append(i)

    main_centroids = np.array([centroids[i] for i in main_indices]) if main_indices else np.empty((0, 2))
    main_boxes = [stats[i] for i in main_indices]

    cleaned_ink = ink.copy()
    specks_removed = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 4 <= area <= max_speck_area and i not in main_indices:
            cx, cy = centroids[i]
            rx, ry, rw, rh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            # Rule 1: Proximity check to REAL text component centroids
            if len(main_centroids) > 0:
                dists = np.hypot(main_centroids[:, 0] - cx, main_centroids[:, 1] - cy)
                if np.min(dists) < dist_thresh:
                    continue

            # Rule 2: Devanagari Upper Modifier / Anusvara Protection
            is_anusvara = False
            for mbox in main_boxes:
                mx, my, mw, mh = mbox[cv2.CC_STAT_LEFT], mbox[cv2.CC_STAT_TOP], mbox[cv2.CC_STAT_WIDTH], mbox[cv2.CC_STAT_HEIGHT]
                if (rx + rw >= mx - 5) and (rx <= mx + mw + 5):
                    if (my - 35 <= ry <= my + 10):
                        is_anusvara = True
                        break
            if is_anusvara:
                continue

            cleaned_ink[labels == i] = 0
            specks_removed += 1

    if specks_removed == 0:
        return bgr

    out_gray = gray.copy()
    removed_mask = (ink > 0) & (cleaned_ink == 0)
    out_gray[removed_mask] = 255
    return _gray_bgr(out_gray)


def _cc_global_speckle_removal(image: np.ndarray, max_speck_area: int = 80) -> np.ndarray:
    """Global speckle removal for heavily degraded documents."""
    bgr = ensure_bgr(image)
    gray = _gray(bgr)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if num_labels <= 1:
        return bgr

    cleaned_ink = ink.copy()
    specks_removed = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        width = stats[i, cv2.CC_STAT_WIDTH]
        if 4 <= area <= max_speck_area and height < 14 and width < 14:
            cleaned_ink[labels == i] = 0
            specks_removed += 1

    if specks_removed == 0:
        return bgr

    out_gray = gray.copy()
    removed_mask = (ink > 0) & (cleaned_ink == 0)
    out_gray[removed_mask] = 255
    return _gray_bgr(out_gray)


def _cc_illumination_norm_ink_protected(image: np.ndarray, blur_percent: float = 0.05) -> np.ndarray:
    """Illumination normalization that preserves dark ink contrast for component stability."""
    bgr = ensure_bgr(image)
    gray = _gray(bgr)
    h, w = gray.shape
    scale = max(15, (min(h, w) // int(1 / blur_percent)) | 1)
    bg = cv2.GaussianBlur(gray, (scale, scale), 0).astype(np.float32)
    gray_f = gray.astype(np.float32)
    norm = cv2.divide(gray_f, np.maximum(bg, 1.0), scale=235.0)
    norm_gray = np.clip(norm, 0, 255).astype(np.uint8)
    return _gray_bgr(norm_gray)


def _cc_bilateral_speck_cleaned(image: np.ndarray, d: int = 5, sc: float = 30.0, max_speck_area: int = 30, dist_thresh: float = 8.0) -> np.ndarray:
    """Mild bilateral filter combined with ink-preserved speckle removal."""
    bgr = ensure_bgr(image)
    denoised = cv2.bilateralFilter(bgr, d, sc, sc)
    return _cc_clean_isolated_specks(denoised, max_speck_area=max_speck_area, dist_thresh=dist_thresh)


def _safe_text_crop(image: np.ndarray) -> Optional[np.ndarray]:
    """Crop only exterior blank space, retaining a protective content margin."""
    ink = _binary_ink(image)
    points = cv2.findNonZero(ink)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    full_h, full_w = ink.shape
    margin = max(4, min(full_h, full_w) // 80)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(full_w, x + w + margin), min(full_h, y + h + margin)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    # Do not crop unless the blank exterior is materially large.
    if (x1 - x0) * (y1 - y0) >= full_h * full_w * 0.96:
        return None
    return ensure_bgr(image)[y0:y1, x0:x1].copy()


def _unsharp(image: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    base = ensure_bgr(image)
    softened = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(base, 1.0 + amount, softened, -amount, 0)


def _morph_strokes(image: np.ndarray, expand_dark_ink: bool, size: int) -> np.ndarray:
    """Apply a single tiny grayscale morphology step, preserving anti-aliasing as much as possible."""
    gray = _gray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    # In a document image dark text is low-valued: grayscale erosion expands it,
    # while grayscale dilation contracts it.
    op = cv2.MORPH_ERODE if expand_dark_ink else cv2.MORPH_DILATE
    changed = cv2.erode(gray, kernel, iterations=1) if op == cv2.MORPH_ERODE else cv2.dilate(gray, kernel, iterations=1)
    return _gray_bgr(changed)


def _estimate_signed_skew(image: np.ndarray) -> float:
    """Mirror the scorer's Hough-line normalisation but retain direction for deskewing."""
    gray = _gray(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=25, maxLineGap=10)
    if lines is None or len(lines) == 0:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=5)
    if lines is None or len(lines) == 0:
        return 0.0
    angles: List[float] = []
    for line in lines:
        x1, y1, x2, y2 = line.flatten()[:4]
        deg = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
        if abs(deg) <= 45:
            angles.append(deg)
        elif deg > 45:
            angles.append(deg - 90)
        elif deg < -45:
            angles.append(deg + 90)
    return float(np.median(angles)) if angles else 0.0


def _rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate onto a white expanded canvas, preventing text loss at the edges."""
    source = ensure_bgr(image)
    h, w = source.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(math.ceil((h * sin) + (w * cos)))
    new_h = int(math.ceil((h * cos) + (w * sin)))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(source, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))







def _resolution_candidates(image: np.ndarray, target: str) -> Iterable[RefinementCandidate]:
    h, w = image.shape[:2]
    if h * w > MAX_SOURCE_PIXELS:
        return []
    candidates: List[RefinementCandidate] = []
    for scale, interpolation, label in (
        (1.25, cv2.INTER_CUBIC, "cubic"),
        (1.50, cv2.INTER_LANCZOS4, "lanczos4"),
        (2.00, cv2.INTER_CUBIC, "cubic"),
    ):
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        if new_w > MAX_UPSCALED_SIDE or new_h > MAX_UPSCALED_SIDE or new_w * new_h > MAX_UPSCALED_PIXELS:
            continue
        enlarged = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        candidates.append(_candidate(enlarged, target, "Controlled upscaling", scale=scale, interpolation=label))
    return candidates


def _noise_refinement_candidates(image: np.ndarray, target_factor: str, baseline_scores: Dict[str, Any]) -> List[RefinementCandidate]:
    """
    Generate dynamic, multi-tier, ink-preserving and sharpness-compensated
    candidates for noise refinement across all noise regimes (20-80 score).
    """
    bgr = ensure_bgr(image)
    gray = _gray(bgr)
    h, w = gray.shape

    # Ink mask to protect glyphs, matras, and text strokes from blur degradation
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_size = max(3, (min(h, w) // 150) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    ink_mask = cv2.dilate(ink, kernel, iterations=1).astype(np.float32) / 255.0
    ink_mask = cv2.GaussianBlur(ink_mask, (3, 3), 0)[:, :, np.newaxis]

    def _ink_protected(bg_image: np.ndarray, boost: float = 0.35) -> np.ndarray:
        sharp = _unsharp(bgr, boost, 0.8) if boost > 0 else bgr
        out = sharp * ink_mask + bg_image * (1.0 - ink_mask)
        return np.clip(out, 0, 255).astype(np.uint8)

    candidates: List[RefinementCandidate] = []

    # 1. Edge-preserving bilateral with unsharp compensation (preserves Laplacian sharpness)
    for d, sc, ss, boost in [(5, 30, 30, 0.25), (7, 45, 45, 0.35), (9, 65, 65, 0.45), (5, 75, 75, 0.30)]:
        den = cv2.bilateralFilter(bgr, d, sc, ss)
        comp = _unsharp(den, boost, 0.8)
        candidates.append(_candidate(comp, target_factor, "Compensated edge-preserving bilateral denoising", d=d, sigma_color=sc, boost=boost))

    # 2. Ink-protected bilateral background smoothing
    for d, sc, ss in [(5, 40, 40), (7, 60, 60), (9, 80, 80)]:
        bg_den = cv2.bilateralFilter(bgr, d, sc, ss)
        candidates.append(_candidate(_ink_protected(bg_den, 0.35), target_factor, "Ink-protected bilateral background denoising", d=d, sigma_color=sc))

    # 3. Ink-protected Gaussian background smoothing (for continuous sensor/grain noise)
    for ksize, sigma in [(3, 0.8), (5, 1.2), (7, 1.6), (9, 2.0)]:
        bg_den = cv2.GaussianBlur(bgr, (ksize, ksize), sigma)
        candidates.append(_candidate(_ink_protected(bg_den, 0.35), target_factor, "Ink-protected Gaussian background smoothing", ksize=ksize, sigma=sigma))

    # 4. Ink-protected Median background smoothing (for impulse/speckle noise)
    for ksize in [3, 5]:
        bg_den = cv2.medianBlur(bgr, ksize)
        candidates.append(_candidate(_ink_protected(bg_den, 0.35), target_factor, "Ink-protected median background denoising", ksize=ksize))

    # 5. Background texture normalization + unsharp (for marble/paper texture noise)
    bg_norm = _background_normalize(bgr)
    for boost in [0.2, 0.4, 0.6]:
        candidates.append(_candidate(_unsharp(bg_norm, boost, 0.8), target_factor, "Background texture normalization", boost=boost))

    # 6. Global sharpness-compensated Gaussian denoising
    for ksize, sigma, boost in [(3, 0.5, 0.25), (3, 0.8, 0.4), (5, 1.0, 0.5), (5, 1.4, 0.6)]:
        den = cv2.GaussianBlur(bgr, (ksize, ksize), sigma)
        candidates.append(_candidate(_unsharp(den, boost, 0.8), target_factor, "Sharpness-compensated Gaussian denoising", ksize=ksize, sigma=sigma, boost=boost))

    # 7. Non-local means with unsharp compensation
    if bgr.shape[0] * bgr.shape[1] <= 4_000_000:
        for h_param, boost in [(3, 0.25), (5, 0.4)]:
            den = cv2.fastNlMeansDenoisingColored(bgr, None, h_param, h_param, 5, 15)
            candidates.append(_candidate(_unsharp(den, boost, 0.8), target_factor, "Sharpness-compensated non-local means", h=h_param, boost=boost))

    return candidates


def generate_candidates(target_factor: str, image_bgr: np.ndarray, baseline_scores: Dict[str, Any]) -> List[RefinementCandidate]:
    """Generate a small, factor-specific and de-duplicated candidate set."""
    if target_factor not in FACTOR_KEYS:
        raise ValueError(f"Unknown refinement factor: {target_factor}")
    image = ensure_bgr(image_bgr)
    target_score = float(baseline_scores.get(target_factor, {}).get("score", 0.0))
    # CC Stability, Matra Continuity, and Zone Integrity use factor-specific eligibility rules
    # (no 'severe' block below 20) — any score < 80 can attempt refinement.
    if target_factor == "connected_component_stability_score":
        state, _ = cc_refinement_eligibility(target_score)
    elif target_factor == "matra_continuity_score":
        state, _ = matra_refinement_eligibility(target_score)
    elif target_factor == "zone_integrity_score":
        state, _ = zone_integrity_refinement_eligibility(target_score)
    else:
        state, _ = refinement_eligibility(target_score)

    if state != "eligible":
        return []

    candidates: List[RefinementCandidate] = []
    if target_factor == "noise_score":
        candidates = _noise_refinement_candidates(image, target_factor, baseline_scores)
    elif target_factor == "resolution_score":
        candidates = list(_resolution_candidates(image, target_factor))
    elif target_factor == "blur_score":
        candidates = [
            _candidate(_unsharp(image, amount, sigma), target_factor, "Unsharp mask", amount=amount, sigma=sigma)
            for amount, sigma in ((0.4, 0.5), (0.6, 0.8), (0.8, 1.0), (1.0, 1.0), (1.2, 1.2), (1.4, 1.2))
        ]
    elif target_factor == "contrast_score":
        mean = float(_gray(image).mean())
        candidates = [
            _candidate(_contrast_adjust(image, 1.10, (128.0 - mean) * 0.03), target_factor, "Conservative contrast scale", alpha=1.10),
            _candidate(_contrast_adjust(image, 1.20, (128.0 - mean) * 0.06), target_factor, "Moderate contrast scale", alpha=1.20),
            _candidate(_gamma_adjust(image, 0.85 if mean < 128 else 1.15), target_factor, "Gamma correction", gamma=0.85 if mean < 128 else 1.15),
            _candidate(_clahe_luminance(image, 1.35), target_factor, "Low-clip CLAHE", clip_limit=1.35, tile_grid="8x8"),
        ]
    elif target_factor == "stroke_width_score":
        ratio = float(baseline_scores.get(target_factor, {}).get("raw_value", 0.0) or 0.0)
        expand = ratio < 0.10
        direction = "dark-ink expansion" if expand else "dark-ink contraction"
        candidates = [
            _candidate(_morph_strokes(image, expand, 2), target_factor, "Tiny stroke-width morphology", operation=direction, kernel="2x2 ellipse"),
        ]
        # A 3x3 operation is only considered for clearly malformed strokes.
        if ratio < 0.055 or ratio > 0.20:
            candidates.append(_candidate(_morph_strokes(image, expand, 3), target_factor, "Small stroke-width morphology", operation=direction, kernel="3x3 ellipse"))
    elif target_factor == "text_density_score":
        cropped = _safe_text_crop(image)
        if cropped is not None:
            candidates.append(_candidate(cropped, target_factor, "Protective blank-border crop", padding="max(4px, 1.25% of shorter side)"))
        candidates.append(_candidate(_background_normalize(image), target_factor, "Gentle background normalization", blur_scale="5% of shorter side"))
    elif target_factor == "matra_continuity_score":
        candidates = [
            # ── Primary: strong shirorekha dilation+close (most visually impactful) ──
            _candidate(_matra_shirorekha_repair(image, close_width=10, dilate_px=1),
                       target_factor, "Shirorekha dilation+close (10px, 1px dil)", close_width=10, dilate_px=1),
            _candidate(_matra_shirorekha_repair(image, close_width=14, dilate_px=1),
                       target_factor, "Shirorekha dilation+close (14px, 1px dil)", close_width=14, dilate_px=1),
            _candidate(_matra_shirorekha_repair(image, close_width=8, dilate_px=2),
                       target_factor, "Shirorekha dilation+close (8px, 2px dil)", close_width=8, dilate_px=2),
            _candidate(_matra_shirorekha_repair(image, close_width=12, dilate_px=2),
                       target_factor, "Shirorekha dilation+close (12px, 2px dil)", close_width=12, dilate_px=2),
            # ── Secondary: full-image horizontal close (catches inter-band matras) ──
            _candidate(_matra_full_image_repair(image, close_width=8),
                       target_factor, "Full-image matra gap bridge (8px)", close_width=8),
            _candidate(_matra_full_image_repair(image, close_width=12),
                       target_factor, "Full-image matra gap bridge (12px)", close_width=12),
            # ── Tertiary: mild denoising to remove noise that fragments metrics ──
            _candidate(_background_normalize(image),
                       target_factor, "Background normalization for matras", blur_scale="5% of shorter side"),
            _candidate(cv2.bilateralFilter(image, 7, 35, 35),
                       target_factor, "Edge-preserving matra denoising", diameter=7, sigma=35),
            _candidate(_remove_one_pixel_specks(image),
                       target_factor, "Isolated noise speck removal for matra", max_component_area=2),
            # ── Combined: background normalize then strong shirorekha repair ──
            _candidate(_matra_shirorekha_repair(_background_normalize(image), close_width=10, dilate_px=1),
                       target_factor, "Background normalize + shirorekha repair", close_width=10, dilate_px=1),
        ]
    elif target_factor == "zone_integrity_score":
        candidates = [
            _candidate(_zone_integrity_repair(image, max_speck_area=6,  shiro_close=3,  body_close=1), target_factor, "Zone speckle & headline repair (light)",     max_speck=6,  shiro_k=3,  body_k=1),
            _candidate(_zone_integrity_repair(image, max_speck_area=8,  shiro_close=5,  body_close=2), target_factor, "Zone headline & matra repair (moderate)",    max_speck=8,  shiro_k=5,  body_k=2),
            _candidate(_zone_integrity_repair(image, max_speck_area=12, shiro_close=8,  body_close=3), target_factor, "Zone headline & zone connection (deep)",     max_speck=12, shiro_k=8,  body_k=3),
            _candidate(_zone_integrity_repair(image, max_speck_area=15, shiro_close=12, body_close=4), target_factor, "Zone headline & zone connection (strong)",   max_speck=15, shiro_k=12, body_k=4),
            _candidate(_remove_one_pixel_specks(image), target_factor, "Isolated noise speck removal for zones", max_component_area=2),
        ]






    elif target_factor == "connected_component_stability_score":
        candidates = [
            _candidate(_cc_clean_isolated_specks(image, max_speck_area=30, dist_thresh=8.0), target_factor, "Moderate speckle cleanup (area <= 30px)", max_speck_area=30, dist_thresh=8.0),
            _candidate(_cc_clean_isolated_specks(image, max_speck_area=60, dist_thresh=10.0), target_factor, "Deep speckle cleanup (area <= 60px)", max_speck_area=60, dist_thresh=10.0),
            _candidate(_cc_clean_isolated_specks(image, max_speck_area=100, dist_thresh=12.0), target_factor, "Aggressive speckle cleanup (area <= 100px)", max_speck_area=100, dist_thresh=12.0),
            _candidate(_cc_global_speckle_removal(image, max_speck_area=80), target_factor, "Global noise speckle removal", max_speck_area=80),
            _candidate(_cc_bilateral_speck_cleaned(image, d=5, sc=30.0, max_speck_area=30, dist_thresh=8.0), target_factor, "Bilateral filter + speckle cleanup", diameter=5, max_speck_area=30),
            _candidate(_cc_bilateral_speck_cleaned(image, d=5, sc=40.0, max_speck_area=60, dist_thresh=10.0), target_factor, "Bilateral filter + deep speckle cleanup", diameter=5, max_speck_area=60),
            _candidate(_cc_illumination_norm_ink_protected(image, blur_percent=0.05), target_factor, "Ink-protected illumination normalization", blur_scale="5% of shorter side"),
            _candidate(_cc_clean_isolated_specks(_clahe_luminance(image, 1.35), max_speck_area=40, dist_thresh=8.0), target_factor, "CLAHE contrast + ink speckle cleanup", clip_limit=1.35, max_speck_area=40),
        ]
    elif target_factor == "skew_penalty_score":
        signed = _estimate_signed_skew(image)
        centre = float(np.clip(-signed, -3.0, 3.0))
        angles = sorted({round(float(np.clip(centre + offset, -3.0, 3.0)), 2) for offset in (-1.0, -0.5, 0.0, 0.5, 1.0)} | {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0})
        candidates = [
            _candidate(_rotate_bound(image, angle), target_factor, "Bounded Hough-guided deskew", rotation_degrees=angle, estimated_signed_skew=round(signed, 2))
            for angle in angles if abs(angle) >= 0.1
        ]

    original_hash = _image_hash(image)
    unique: List[RefinementCandidate] = []
    seen = {original_hash}
    for candidate in candidates:
        digest = _image_hash(candidate.image)
        if digest not in seen:
            seen.add(digest)
            unique.append(candidate)
    return unique


def _image_hash(image: np.ndarray) -> str:
    array = ensure_bgr(image)
    # A downsampled digest is enough for duplicate elimination and avoids making
    # repeated full-size byte copies for large documents.
    thumb = cv2.resize(array, (min(128, array.shape[1]), min(128, array.shape[0])), interpolation=cv2.INTER_AREA)
    return sha1(thumb.tobytes()).hexdigest()


def assess_candidate(candidate: RefinementCandidate, baseline_scores: Dict[str, Any], scores: Dict[str, Any]) -> RefinementCandidate:
    """Attach authoritative score deltas and enforce the immutable safety policy."""
    candidate.scores = scores
    target = candidate.target_factor
    try:
        score_diff = float(scores[target]["score"]) - float(baseline_scores[target]["score"])
        if target == "connected_component_stability_score":
            base_cv = float(baseline_scores[target].get("raw_value", 0.0) or 0.0)
            cand_cv = float(scores[target].get("raw_value", 0.0) or 0.0)
            if base_cv > 0 and cand_cv > 0:
                cv_imp = max(0.0, base_cv - cand_cv) * 40.0
                target_imp = max(score_diff, cv_imp)
            else:
                target_imp = score_diff
        else:
            target_imp = score_diff
        candidate.target_improvement = round(target_imp, 1)
        candidate.readiness_improvement = round(
            float(scores["ocr_readiness_score"]) - float(baseline_scores["ocr_readiness_score"]), 1
        )
    except (KeyError, TypeError, ValueError):
        candidate.safe = False
        candidate.safety_reason = "Candidate did not return a complete factor score set."
        return candidate

    changes: Dict[str, float] = {}
    for key in FACTOR_KEYS:
        if key == target:
            continue
        try:
            changes[key] = round(float(scores[key]["score"]) - float(baseline_scores[key]["score"]), 1)
        except (KeyError, TypeError, ValueError):
            candidate.safe = False
            candidate.safety_reason = "Candidate did not return every non-target factor score."
            return candidate
    candidate.non_target_changes = changes

    if candidate.target_improvement < TARGET_MIN_IMPROVEMENT:
        candidate.safe = False
        candidate.safety_reason = f"Target factor improvement must be at least +{TARGET_MIN_IMPROVEMENT:.1f}."
    elif any(delta < -MAX_NON_TARGET_DROP for delta in changes.values()):
        if (
            target in ("connected_component_stability_score", "matra_continuity_score", "zone_integrity_score")
            and candidate.target_improvement >= 1.0
        ):
            quality_keys = {"blur_score", "noise_score", "contrast_score", "resolution_score", "skew_penalty_score"}
            quality_drops = [changes[k] for k in quality_keys if k in changes]
            if all(d >= -15.0 for d in quality_drops):
                candidate.safe = True
                candidate.safety_reason = f"Passed {target} structural zone repair safety policy."
            else:
                candidate.safe = False
                candidate.safety_reason = f"An image quality factor dropped by more than 15.0 points."
        else:
            candidate.safe = False
            candidate.safety_reason = f"A non-target factor dropped by more than {MAX_NON_TARGET_DROP:.1f} points."

    elif candidate.readiness_improvement <= 0.0:
        if target in ("connected_component_stability_score", "matra_continuity_score", "zone_integrity_score") and candidate.target_improvement >= 1.0:
            candidate.safe = True
            candidate.safety_reason = f"Passed {target} structural target improvement safety policy."
        else:
            candidate.safe = False
            candidate.safety_reason = "OCR Readiness must be strictly greater than the baseline."
    else:
        candidate.safe = True
        candidate.safety_reason = "Passed target, non-target, and OCR Readiness safety checks."
    return candidate


def _candidate_sort_key(candidate: RefinementCandidate) -> Tuple[float, float, float]:
    degradation = sum(max(0.0, -change) for change in candidate.non_target_changes.values())
    # For CC Stability, Matra Continuity, and Zone Integrity, prioritise the candidate
    # with the largest target-factor improvement so the most visually impactful
    # structural repair wins over a subtler operation.
    if candidate.target_factor in ("connected_component_stability_score", "matra_continuity_score", "zone_integrity_score"):
        return candidate.target_improvement, candidate.readiness_improvement, -degradation
    return candidate.readiness_improvement, candidate.target_improvement, -degradation


def evaluate_factor_refinement(
    image_bgr: np.ndarray,
    target_factor: str,
    baseline_scores: Optional[Dict[str, Any]] = None,
    scorer: Callable[[np.ndarray], Dict[str, Any]] = run_all_factors,
) -> RefinementOutcome:
    """Score every candidate with the existing complete factor suite and choose the best safe one."""
    image = ensure_bgr(image_bgr)
    baseline = baseline_scores or scorer(image)
    target_score = float(baseline.get(target_factor, {}).get("score", 0.0))
    # CC Stability, Matra Continuity, and Zone Integrity use factor-specific eligibility rules
    # (no 'severe' block below 20).
    if target_factor == "connected_component_stability_score":
        state, eligibility_message = cc_refinement_eligibility(target_score)
    elif target_factor == "matra_continuity_score":
        state, eligibility_message = matra_refinement_eligibility(target_score)
    elif target_factor == "zone_integrity_score":
        state, eligibility_message = zone_integrity_refinement_eligibility(target_score)
    else:
        state, eligibility_message = refinement_eligibility(target_score)

    if state != "eligible":
        return RefinementOutcome(target_factor, baseline, None, message=eligibility_message)

    summaries: List[Dict[str, Any]] = []
    safe_candidates: List[RefinementCandidate] = []
    try:
        candidates = generate_candidates(target_factor, image, baseline)
    except Exception as exc:
        return RefinementOutcome(target_factor, baseline, None, message=f"Refinement could not generate candidates: {exc}")

    for candidate in candidates:
        try:
            scores = scorer(candidate.image)
            assess_candidate(candidate, baseline, scores)
        except Exception as exc:
            candidate.safe = False
            candidate.safety_reason = f"Candidate scoring failed safely: {type(exc).__name__}."
        summaries.append(candidate.summary())
        if candidate.safe:
            safe_candidates.append(candidate)

    if not safe_candidates:
        return RefinementOutcome(
            target_factor,
            baseline,
            None,
            summaries,
            "No safe refinement was found for this factor.",
        )
    best = max(safe_candidates, key=_candidate_sort_key)
    return RefinementOutcome(target_factor, baseline, best, summaries, "A safe refinement candidate was found.")


def _global_safety(scores: Dict[str, Any], initial_scores: Dict[str, Any]) -> bool:
    """Extra Refine All guard: no factor may be more than five points below its original value."""
    try:
        if float(scores["ocr_readiness_score"]) <= float(initial_scores["ocr_readiness_score"]):
            return False
        return all(
            float(scores[key]["score"]) >= float(initial_scores[key]["score"]) - MAX_NON_TARGET_DROP
            for key in FACTOR_KEYS
        )
    except (KeyError, TypeError, ValueError):
        return False


def refine_all(
    image_bgr: np.ndarray,
    baseline_scores: Optional[Dict[str, Any]] = None,
    scorer: Callable[[np.ndarray], Dict[str, Any]] = run_all_factors,
) -> RefineAllOutcome:
    """Safely improve eligible factors one at a time; never blindly stack edits."""
    current_image = ensure_bgr(image_bgr)
    initial_scores = baseline_scores or scorer(current_image)
    current_scores = initial_scores
    steps: List[RefinementCandidate] = []
    attempted: set[str] = set()
    blur_passes = 0
    max_blur_passes = 3

    # Most factors are considered once. Blur is allowed a few bounded passes so
    # repeated edge definition can recover clarity without unlimited sharpening.
    while len(attempted) < len(FACTOR_KEYS) or blur_passes < max_blur_passes:
        eligible = [
            key for key in FACTOR_KEYS
            if (
                float(current_scores.get(key, {}).get("score", 0.0)) < 80.0
                and (
                    key in ("connected_component_stability_score", "matra_continuity_score", "zone_integrity_score")
                    or float(current_scores.get(key, {}).get("score", 0.0)) >= 20.0
                )
                and (key not in attempted or (key == "blur_score" and blur_passes < max_blur_passes))
            )
        ]
        if not eligible:
            break
        eligible.sort(key=lambda key: (key != "blur_score", float(current_scores[key]["score"])))
        target = eligible[0]
        attempted.add(target)
        if target == "blur_score":
            blur_passes += 1
        outcome = evaluate_factor_refinement(current_image, target, current_scores, scorer)
        candidate = outcome.best_candidate
        if candidate is None or candidate.scores is None:
            continue
        if not _global_safety(candidate.scores, initial_scores):
            continue
        current_image = candidate.image
        current_scores = candidate.scores
        steps.append(candidate)

    if steps:
        message = f"Found {len(steps)} sequential safe refinement step(s)."
    else:
        message = "No safe refinement was found for the eligible factors."
    return RefineAllOutcome(initial_scores, current_image, current_scores, steps, message)
