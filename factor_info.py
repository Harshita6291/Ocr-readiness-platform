"""
Educational content for the About Factor section.
Each entry has: definition, importance, formula, ocr_impact, ideal_range.
Updated to 100% match the algorithmic implementation in factors.py.
"""

FACTOR_INFO = {
    "noise_score": {
        "display_name": "Noise Score",
        "owner": "Yash",
        "definition": (
            "Image noise refers to random high-frequency variations in brightness or color, "
            "typically caused by sensor ISO grain, low lighting, or compression artifacts. "
            "To prevent sharp, dense text from being penalized as noise, evaluation isolates "
            "the flat background regions outside character strokes."
        ),
        "importance": (
            "Noise introduces false edge gradients and degrades character contours. In Devanagari OCR, "
            "background grain can be misread as diacritics (like anusvara dots 'ं') or cause punctuation marks "
            "and thin strokes to break, generating character substitution errors."
        ),
        "formula": (
            "Flat-Region High-Frequency Residual Method:\n"
            "1. Detect character stroke edges via Canny (50, 150).\n"
            "2. Dilate edge mask (7×7 kernel) to exclude text strokes and margins.\n"
            "3. Compute Gaussian residual on grayscale image: residual = gray − GaussianBlur(gray, 5×5).\n"
            "4. Compute background noise standard deviation: σ = std(residual[flat_mask]).\n"
            "5. Score = clamp(100.0 − σ / 0.30, 0, 100)."
        ),
        "ocr_impact": (
            "High noise levels (σ > 15) sharply reduce OCR confidence and character accuracy. "
            "Isolating noise to background regions ensures clean text scans maintain top scores "
            "while recommending targeted denoising (e.g. bilateral or median filtering) when needed."
        ),
        "ideal_range": "Score ≥ 70 (σ ≤ 9.0). Excellent: Score ≥ 81 (σ ≤ 5.7).",
    },

    "resolution_score": {
        "display_name": "Resolution Score",
        "owner": "Yash",
        "definition": (
            "Resolution describes the total pixel detail available across the image, "
            "measured in total Megapixels (MP = height × width / 1,000,000). Rather than evaluating "
            "only one dimension, MP captures the true 2D character detail."
        ),
        "importance": (
            "OCR engines require sufficient pixel sampling to distinguish visually similar characters "
            "(such as 'rn' vs 'm', '0' vs 'O', or Devanagari matras like 'ि' vs 'ी'). Below standard DPI thresholds, "
            "fine glyph features blur into single pixel clusters."
        ),
        "formula": (
            "Megapixel OCR Calibration Method:\n"
            "1. Compute total image megapixels: MP = (height × width) / 1,000,000.\n"
            "2. If MP ≥ 2.0 (standard ~300 DPI document): Score = 100.\n"
            "3. If 1.0 ≤ MP < 2.0: Score = 70 + (MP − 1.0) × 30.\n"
            "4. If 0.3 ≤ MP < 1.0: Score = 35 + ((MP − 0.3) / 0.7) × 35.\n"
            "5. If MP < 0.3: Score = (MP / 0.3) × 35."
        ),
        "ocr_impact": (
            "Tesseract and modern OCR pipelines are trained predominantly on 300 DPI document imagery. "
            "Resolutions below 1.0 MP cause character segmentation errors and broken loops. Beyond 2.0 MP, "
            "OCR accuracy reaches optimal stability with diminishing returns."
        ),
        "ideal_range": "Score ≥ 70 (≥ 1.0 MP, ~300 DPI equivalent). Excellent: Score ≥ 81 (≥ 1.37 MP).",
    },

    "blur_score": {
        "display_name": "Blur Score",
        "owner": "Mansi",
        "definition": (
            "Blur represents loss of edge sharpness caused by optical defocus, camera motion shake, "
            "or lossy compression. It manifests as soft, wide transition gradients between text strokes "
            "and the surrounding paper background."
        ),
        "importance": (
            "OCR binarization algorithms rely on sharp, localized gradient peaks to define stroke perimeters. "
            "Blur disperses edge energy, causing thin strokes to disappear and adjacent characters to merge."
        ),
        "formula": (
            "Laplacian Variance with Logarithmic Document Scaling:\n"
            "1. Convert to grayscale and apply mild 3×3 Gaussian smoothing.\n"
            "2. Compute Laplacian operator (second spatial derivative): ∇²I = cv2.Laplacian(gray, CV_64F).\n"
            "3. Calculate edge response variance: lap_var = var(∇²I).\n"
            "4. Perceptual log-scaling: Score = clamp(((ln(1 + lap_var) − ln(1 + 3)) / (ln(1 + 600) − ln(1 + 3))) × 100, 0, 100)."
        ),
        "ocr_impact": (
            "Blur is consistently the single strongest predictor of OCR failure. Soft edges cause severe "
            "under- or over-binarization, turning complex characters into unreadable blobs. Sharpening filters "
            "or unsharp masking are recommended when blur score drops below 60."
        ),
        "ideal_range": "Score ≥ 70 (Laplacian variance ≥ 75). Excellent: Score ≥ 81 (variance ≥ 175).",
    },

    "contrast_score": {
        "display_name": "Contrast Score",
        "owner": "Mansi",
        "definition": (
            "Contrast measures the luminance separation between foreground text ink and the background paper. "
            "To prevent document layout density from skewing results, an Otsu-partitioned intensity difference "
            "is computed directly between ink and background pixel clusters."
        ),
        "importance": (
            "When text contrast is low, ink intensity is nearly indistinguishable from page shading or paper tint. "
            "Adaptive and global thresholding both fail, resulting in dropped letters, broken words, or heavy pepper noise."
        ),
        "formula": (
            "Otsu Foreground/Background Intensity Separation Method:\n"
            "1. Convert image to grayscale.\n"
            "2. Apply Otsu automatic thresholding to segment foreground (ink) and background (paper).\n"
            "3. Compute mean luminance difference: contrast = mean(bg_pixels) − mean(fg_pixels).\n"
            "4. Calibrated score: Score = clamp(((contrast − 20) / (180 − 20)) × 100, 0, 100)."
        ),
        "ocr_impact": (
            "High contrast (intensity separation > 130) guarantees clean character binarization. "
            "Separation below 60 causes characters to fade into the background during thresholding. "
            "Contrast enhancement (like CLAHE) reliably restores readability for low-contrast scans."
        ),
        "ideal_range": "Score ≥ 70 (contrast separation ≥ 132). Excellent: Score ≥ 81 (separation ≥ 150).",
    },

    "stroke_width_score": {
        "display_name": "Stroke Width Score",
        "owner": "Vivek",
        "definition": (
            "Stroke width evaluates the physical thickness of text lines relative to character height "
            "(ratio R = W_stroke / H_char). Evaluating relative proportions rather than fixed pixel counts "
            "ensures fair scoring across all font sizes, headings, and capture scales."
        ),
        "importance": (
            "Very thin relative strokes (R < 0.05) fragment under standard binarization, producing broken glyphs. "
            "Very heavy strokes (R > 0.22) fill in internal character loops (counters) in letters like 'e', 'a', 'म', 'ब' "
            "and cause adjacent glyphs to fuse together."
        ),
        "formula": (
            "Typographic Ratio & Plateau Method:\n"
            "1. Binarize text using Otsu inversion (THRESH_BINARY_INV).\n"
            "2. Filter valid character bounding boxes via cv2.connectedComponentsWithStats to find median character height (H_char).\n"
            "3. Compute Euclidean Distance Transform on text pixels and extract morphological skeleton for median stroke width (W_stroke).\n"
            "4. Calculate stroke-to-height ratio: R = W_stroke / H_char (ideal typographic ratio ~0.10–0.12).\n"
            "5. Piecewise scoring curve:\n"
            "   • If 0.08 ≤ R ≤ 0.15: Score = 90 + 10 × (1 − |R − 0.115| / 0.035)  [Typographic plateau]\n"
            "   • If R < 0.08: Score = 90 × (R / 0.08)^1.3  [Thin / broken stroke falloff]\n"
            "   • If R > 0.15: Score = 90 × exp(−0.5 × ((R − 0.15) / 0.11)^1.7)  [Heavy / bold text falloff]"
        ),
        "ocr_impact": (
            "Optimal OCR recognition occurs when stroke width occupies 8%–15% of character height. "
            "The calibrated plateau prevents standard bold or light typefaces from being unfairly penalized, "
            "while strictly flagging severely eroded or over-inked text."
        ),
        "ideal_range": "Score ≥ 70 (ratio R ≈ 0.06–0.20). Excellent: Score ≥ 81 (ratio R ≈ 0.08–0.15).",
    },

    "text_density_score": {
        "display_name": "Text Density Score",
        "owner": "Vivek",
        "definition": (
            "Text density measures the proportion of image pixels occupied by dark foreground text ink "
            "versus the total document surface area. It evaluates page layout utilization and content distribution."
        ),
        "importance": (
            "Extremely sparse images (< 3% coverage) indicate excessive margins or empty borders that degrade resolution efficiency. "
            "Overly packed images (> 45% coverage) suffer from touching lines, dense tabular clutter, or inverted thresholding."
        ),
        "formula": (
            "Gaussian Layout Density Method:\n"
            "1. Binarize image using Otsu thresholding.\n"
            "2. Compute coverage percentage: density_pct = (text_pixels / total_pixels) × 100.\n"
            "3. Evaluate using Gaussian bell curve centered at 20% coverage (σ = 15%):\n"
            "   Score = clamp(100.0 × exp(−0.5 × ((density_pct − 20.0) / 15.0)²), 0, 100)."
        ),
        "ocr_impact": (
            "OCR page layout analysis engines operate with highest accuracy on balanced 10%–30% density documents. "
            "Cropping tight text boundaries around sparse images maximizes character pixel detail and eliminates border noise."
        ),
        "ideal_range": "Score ≥ 70 (density ≈ 7%–33%). Excellent: Score ≥ 81 (density ≈ 10%–30%).",
    },

    "matra_continuity_score": {
        "display_name": "Matra Continuity Score",
        "owner": "Krish",
        "definition": (
            "In Devanagari script, characters in a word hang from a continuous horizontal headline "
            "known as the Shirorekha (शिरोरेखा). The Matra Continuity Score evaluates the continuity of this "
            "headline across character bodies within words, alongside upper and lower vowel modifier preservation."
        ),
        "importance": (
            "Devanagari OCR engines (including Tesseract's Devanagari LSTM engine) depend on the unbroken Shirorekha "
            "for word boundary segmentation. Gaps inside words cause the OCR engine to misread a single word as multiple "
            "disjointed characters or spurious symbols."
        ),
        "formula": (
            "Word-Span Shirorekha & Modifier Method:\n"
            "1. Detect horizontal text bands via row projection; merge adjacent bands within 15px to preserve unified line structure.\n"
            "2. Locate Shirorekha peak projection row in the upper 15%–45% zone of each line.\n"
            "3. Identify word columns via character body projection in the middle zone.\n"
            "4. Compute Shirorekha coverage across words and unbroken run length factor:\n"
            "   run_factor = min(1.0, mean_run / 18.0)\n"
            "   SCS = clamp(coverage × 60.0 × run_factor + min(40.0, mean_run × 1.5), 0, 100)\n"
            "5. Combine with Upper Matra Visibility (MVS_upper), Lower Matra Visibility, Baseline Stability (BS), and Run-Length Regularity (RLR):\n"
            "   Line MCS = 0.65 × SCS + 0.15 × MVS_upper + 0.10 × MVS_lower + 0.05 × BS + 0.05 × RLR."
        ),
        "ocr_impact": (
            "Devanagari OCR word accuracy drops 25%–50% when the Shirorekha is broken. Word-span continuity "
            "accurately distinguishes between expected word spaces and damaging intra-word headline breaks."
        ),
        "ideal_range": "Score ≥ 70 (Continuous Shirorekha across word spans). Excellent: Score ≥ 81.",
    },

    "zone_integrity_score": {
        "display_name": "Zone Integrity Score",
        "owner": "Krish",
        "definition": (
            "Devanagari text lines are structurally partitioned into vertical zones:\n"
            "• Upper Zone (0%–22%): ascenders and vowel signs (े, ै, ो, ौ, ं, ँ).\n"
            "• Shirorekha Zone (22%–36%): the continuous horizontal headline.\n"
            "• Middle Zone (36%–72%): main consonant bodies and conjuncts.\n"
            "• Lower Zone (72%–100%): descending vowel modifiers (ु, ू, ृ, ्).\n"
            "Zone Integrity verifies that all modifier zones remain intact, distinct, and free of degradation."
        ),
        "importance": (
            "Devanagari vowels are represented by modifying diacritics placed above and below the consonant body. "
            "If modifier zones are clipped by page margins, smudged, or eroded into noise specks, the phonetic "
            "identity and meaning of the words are completely lost."
        ),
        "formula": (
            "Unified Structural Zone Analysis Method:\n"
            "1. Detect text lines with 15px band merging to retain upper and lower modifiers in unified bands.\n"
            "2. Partition line into 4 vertical zones: Upper (0–22%), Shiro (22–36%), Middle (36–72%), Lower (72–100%).\n"
            "3. Evaluate each zone for component health:\n"
            "   • clean_ratio = components with area ≥ 8 px / total components\n"
            "   • noise_ratio = tiny noise specks < 6 px / total components\n"
            "   • cov = standard deviation / mean of distance transform values in zone\n"
            "   • Zone Score = clamp(100 × clean_ratio − 40 × noise_ratio − max(0, (cov − 0.4) × 50), 10, 100)\n"
            "4. Combine zones with Pal-Chaudhuri weights: ZIS = 0.25 × Z_upper + 0.30 × Z_shiro + 0.30 × Z_mid + 0.15 × Z_lower."
        ),
        "ocr_impact": (
            "Clipped or damaged modifier zones cause OCR engines to misrecognize or omit vowels entirely. "
            "Ensuring intact zones preserves full lexical accuracy for Indian languages."
        ),
        "ideal_range": "Score ≥ 70 (All modifier zones intact and well-separated). Excellent: Score ≥ 81.",
    },

    "connected_component_stability_score": {
        "display_name": "Connected Component Stability Score",
        "owner": "Tanusha",
        "definition": (
            "A connected component represents a contiguous cluster of foreground pixels — typically "
            "a character, conjunct, or modifier. CC Stability measures the size consistency of these components "
            "using the Coefficient of Variation (CV = std / mean of component areas)."
        ),
        "importance": (
            "Uniform character sizes indicate clean print quality and proper segmentation. Heavy speckle noise, "
            "fragmented glyphs, or severe ink bleeding create erratic component sizes that disrupt OCR line analysis."
        ),
        "formula": (
            "Typographic Component Variation Method:\n"
            "1. Binarize image (Otsu) and identify connected components via cv2.connectedComponentsWithStats.\n"
            "2. Filter out tiny dust specks (< 4 px area) and page background.\n"
            "3. Compute area mean and standard deviation: CV = std(areas) / mean(areas).\n"
            "4. Typographically calibrated curve (accounting for natural conjunct and matra size variations):\n"
            "   Score = clamp(100.0 − max(0.0, CV − 0.35) × 45.0, 0, 100)."
        ),
        "ocr_impact": (
            "Natural printed Devanagari text exhibits CV values between 0.7 and 1.1 due to legitimate ligature variations. "
            "Values exceeding 1.8 indicate severe pepper noise or shattered characters that cause false line breaks."
        ),
        "ideal_range": "Score ≥ 70 (CV ≤ 1.0). Excellent: Score ≥ 81 (CV ≤ 0.77).",
    },

    "skew_penalty_score": {
        "display_name": "Skew Penalty Score",
        "owner": "Tanusha",
        "definition": (
            "Skew represents the rotational tilt angle of text baselines and headlines relative to the horizontal axis. "
            "Evaluation uses the Hough Line Transform to detect actual linear text baseline orientations."
        ),
        "importance": (
            "OCR engines segment text assuming horizontal lines. Even small skew angles cause character baselines "
            "to drift across lines, resulting in merged lines, chopped words, and jumbled reading order."
        ),
        "formula": (
            "Progressive Hough Line Transform Method:\n"
            "1. Compute Canny edge map of grayscale image.\n"
            "2. Detect text line segments via Probabilistic Hough Lines: cv2.HoughLinesP(edges, threshold=40, minLineLength=25, maxLineGap=10).\n"
            "3. Calculate segment angles θ = arctan2(Δy, Δx) and normalize to [-45°, 45°].\n"
            "4. Determine median text line skew angle: skew_deg = |median(θ)|.\n"
            "5. Compute penalty score: Score = clamp(100.0 − skew_deg × 6.0, 0, 100)."
        ),
        "ocr_impact": (
            "Angles exceeding 3° reduce Tesseract line segmentation reliability. Skew beyond 10° produces severe "
            "character dropouts. When skew penalty score drops below 60, deskewing is strongly recommended prior to OCR."
        ),
        "ideal_range": "Score ≥ 70 (skew angle ≤ 5.0°). Excellent: Score ≥ 81 (skew angle ≤ 3.1°).",
    },
}
