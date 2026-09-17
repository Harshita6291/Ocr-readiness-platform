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
        "method": "Safe component cleanup",
        "details": "Tests mild denoising, one-pixel speck removal, and tiny morphology. Devanagari modifiers and valid small components are retained.",
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


def generate_candidates(target_factor: str, image_bgr: np.ndarray, baseline_scores: Dict[str, Any]) -> List[RefinementCandidate]:
    """Generate a small, factor-specific and de-duplicated candidate set."""
    if target_factor not in FACTOR_KEYS:
        raise ValueError(f"Unknown refinement factor: {target_factor}")
    image = ensure_bgr(image_bgr)
    target_score = float(baseline_scores.get(target_factor, {}).get("score", 0.0))
    state, _ = refinement_eligibility(target_score)
    if state != "eligible":
        return []

    candidates: List[RefinementCandidate] = []
    if target_factor == "noise_score":
        candidates = [
            _candidate(cv2.GaussianBlur(image, (3, 3), 0.45), target_factor, "Mild Gaussian denoising", kernel=3, sigma=0.45),
            _candidate(cv2.medianBlur(image, 3), target_factor, "Mild median denoising", kernel=3),
            _candidate(cv2.bilateralFilter(image, 5, 22, 22), target_factor, "Edge-preserving bilateral denoising", diameter=5, sigma_color=22, sigma_space=22),
            _candidate(cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21), target_factor, "Low-strength non-local means denoising", h=3, template=7, search=21),
        ]
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
            _candidate(_background_normalize(image), target_factor, "Background normalization for matras", blur_scale="5% of shorter side"),
            _candidate(cv2.bilateralFilter(image, 5, 18, 18), target_factor, "Mild edge-preserving matra denoising", diameter=5),
            _candidate(_horizontal_close(image, 2), target_factor, "Tiny horizontal matra gap closure", kernel="2x1", max_gap="1px"),
        ]
    elif target_factor == "zone_integrity_score":
        candidates = [
            _candidate(_background_normalize(image), target_factor, "Zone-preserving background normalization", blur_scale="5% of shorter side"),
            _candidate(cv2.medianBlur(image, 3), target_factor, "Mild zone-preserving denoising", kernel=3),
            _candidate(_remove_one_pixel_specks(image), target_factor, "Isolated one-pixel speck cleanup", max_component_area=2),
            _candidate(_horizontal_close(image, 2), target_factor, "Tiny structural horizontal close", kernel="2x1"),
        ]
    elif target_factor == "connected_component_stability_score":
        candidates = [
            _candidate(cv2.medianBlur(image, 3), target_factor, "Mild component-preserving denoising", kernel=3),
            _candidate(_remove_one_pixel_specks(image), target_factor, "Isolated one-pixel speck cleanup", max_component_area=2),
            _candidate(_horizontal_close(image, 2), target_factor, "Tiny component gap closure", kernel="2x1", max_gap="1px"),
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
        candidate.target_improvement = round(float(scores[target]["score"]) - float(baseline_scores[target]["score"]), 1)
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
        candidate.safe = False
        candidate.safety_reason = f"A non-target factor dropped by more than {MAX_NON_TARGET_DROP:.1f} points."
    elif candidate.readiness_improvement <= 0.0:
        candidate.safe = False
        candidate.safety_reason = "OCR Readiness must be strictly greater than the baseline."
    else:
        candidate.safe = True
        candidate.safety_reason = "Passed target, non-target, and OCR Readiness safety checks."
    return candidate


def _candidate_sort_key(candidate: RefinementCandidate) -> Tuple[float, float, float]:
    degradation = sum(max(0.0, -change) for change in candidate.non_target_changes.values())
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

    # Each factor is considered once per run.  A failed factor is not retried on
    # a later image because that would multiply expensive complete rescoring.
    while len(attempted) < len(FACTOR_KEYS):
        eligible = [
            key for key in FACTOR_KEYS
            if key not in attempted and 20.0 <= float(current_scores.get(key, {}).get("score", 0.0)) < 80.0
        ]
        if not eligible:
            break
        eligible.sort(key=lambda key: float(current_scores[key]["score"]))
        target = eligible[0]
        attempted.add(target)
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
