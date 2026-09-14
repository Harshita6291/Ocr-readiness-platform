"""
Detailed human-readable descriptions for every factor at every score range.
Called after scoring to give the user a clear explanation of what the score means.
"""

def get_factor_description(factor_key: str, score: float) -> str:
    """
    Returns a detailed description based on the factor and its score range.
    Covers all 4 ranges: Excellent (81-100), Good (61-80), Average (41-60), Poor (0-40).
    """

    descs = {

        "noise_score": {
            "excellent": (
                "✅ The image has very low noise. The pixel values are smooth and consistent, "
                "with minimal random speckles or grain. This means character edges are clean "
                "and well-defined. OCR engines will have no trouble distinguishing strokes "
                "from the background. No denoising is needed."
            ),
            "good": (
                "🟡 The image has acceptable noise levels. There is a small amount of grain "
                "or speckle in the pixel values, but it is not severe enough to significantly "
                "affect OCR accuracy. Most characters should still be read correctly. "
                "Optional: a light Gaussian blur or median filter could marginally improve results."
            ),
            "average": (
                "🟠 Moderate noise is detected in this image. Random pixel variations are "
                "present at a level that can interfere with thin strokes and small characters. "
                "OCR may produce occasional misreadings, especially for similar-looking characters "
                "like 'l', 'i', '1'. Recommended: apply a denoising filter (e.g., cv2.fastNlMeansDenoising) "
                "before running OCR."
            ),
            "poor": (
                "🔴 High noise detected. The image is heavily speckled or grainy — this could "
                "be due to a low-quality scan, poor lighting, or a compressed image with artifacts. "
                "Noise at this level will significantly degrade OCR accuracy as it creates false "
                "edges and breaks character strokes. Strong denoising is required before OCR."
            ),
        },

        "resolution_score": {
            "excellent": (
                "✅ The image has high resolution with sufficient pixel density for OCR. "
                "Characters are large enough in pixel terms for the OCR engine to clearly "
                "see internal details like serifs, dots, and thin strokes. This resolution "
                "is equivalent to or better than the 300 DPI standard recommended for OCR scanning."
            ),
            "good": (
                "🟡 The image resolution is adequate for OCR. Most characters should be "
                "recognizable, though very small fonts or detailed characters might be "
                "slightly affected. For best results, ensure the image is at least 300 DPI "
                "if rescanning is possible."
            ),
            "average": (
                "🟠 The image resolution is below the recommended level. Characters may "
                "appear pixelated when zoomed in, and fine details like punctuation marks, "
                "dots above letters, and thin strokes may be lost. OCR accuracy will be "
                "reduced especially for small text. Consider rescanning at a higher DPI."
            ),
            "poor": (
                "🔴 Very low resolution detected. The image does not have enough pixels to "
                "represent characters clearly. OCR will struggle to distinguish between "
                "similar-looking characters and may produce mostly incorrect output. "
                "Rescan the document at minimum 300 DPI, ideally 600 DPI."
            ),
        },

        "blur_score": {
            "excellent": (
                "✅ The image is sharp with excellent edge definition. Character boundaries "
                "are crisp and clearly distinguishable from the background. The Laplacian "
                "variance is high, confirming strong high-frequency detail in the image. "
                "OCR engines will accurately detect stroke edges and character shapes."
            ),
            "good": (
                "🟡 The image is reasonably sharp. There is a slight softness to edges but "
                "character outlines are still distinguishable. Most text should be read "
                "correctly. If the image was photographed rather than scanned, "
                "repositioning the camera with better focus could improve results."
            ),
            "average": (
                "🟠 Noticeable blur is present in this image. Character edges are soft and "
                "strokes bleed into the background. This commonly happens with camera-captured "
                "documents, camera shake, or out-of-focus scanning. OCR accuracy will be "
                "reduced especially for dense text. Try applying unsharp masking or re-capturing "
                "the image with a steady hand and proper focus."
            ),
            "poor": (
                "🔴 The image is severely blurred. Character shapes are not well-defined and "
                "strokes merge with each other and the background. The Laplacian variance is "
                "very low, indicating almost no sharp edges. OCR on this image will produce "
                "very unreliable results. The image needs to be recaptured with proper focus "
                "or a deblurring algorithm should be applied."
            ),
        },

        "contrast_score": {
            "excellent": (
                "✅ Excellent contrast between text and background. The intensity difference "
                "between dark ink and light paper is very high, making it trivial for OCR "
                "to separate foreground (text) from background. This is the ideal condition "
                "for OCR — black text on white paper with no fading or shadows."
            ),
            "good": (
                "🟡 Good contrast levels. Text is clearly darker than the background with "
                "a comfortable separation. Minor variations like slight yellowing of paper "
                "or light pencil marks won't significantly affect OCR performance."
            ),
            "average": (
                "🟠 Moderate contrast detected. The difference between text and background "
                "brightness is smaller than ideal. This can happen with faded ink, colored "
                "paper, shadows across the document, or photocopies of photocopies. "
                "OCR may struggle with lighter characters. Try increasing contrast using "
                "adaptive histogram equalization (CLAHE) or brightness/contrast adjustment."
            ),
            "poor": (
                "🔴 Very low contrast. The text and background have similar brightness levels, "
                "making it very difficult for OCR to locate and read characters. This is "
                "commonly caused by extremely faded documents, pencil writing on light paper, "
                "or bad lighting during scanning. Apply strong contrast enhancement or "
                "adaptive thresholding (Otsu's method) before running OCR."
            ),
        },

        "stroke_width_score": {
            "excellent": (
                "✅ Stroke width is in the optimal range (approximately 10–12% of character height). "
                "Characters have the ideal thickness — not too thin to break under noise, and not "
                "too thick to cause character fills or merges. OCR models are trained on text with "
                "this stroke profile and will perform optimally."
            ),
            "good": (
                "🟡 Stroke width is slightly outside the ideal range but still acceptable (around "
                "8–9% or 13–15% of character height). Characters may be slightly thinner or thicker "
                "than optimal, but OCR should still produce good results for most fonts."
            ),
            "average": (
                "🟠 Stroke width is noticeably outside the optimal range. If strokes are too "
                "thin (< 6% of character height), noise or compression can break characters apart. If too thick "
                "(> 18% of character height), character loops ('e', 'a', 'o') may fill in and merge. "
                "Consider adjusting the scan resolution or binarization threshold."
            ),
            "poor": (
                "🔴 Stroke width is far outside the optimal range (< 4% or > 22% of character height). "
                "Text strokes are either too faint and broken or heavily over-inked and blobbed together. "
                "Both conditions severely impact OCR segmentation. Adjust scan resolution or capture quality."
            ),
        },

        "text_density_score": {
            "excellent": (
                "✅ Text density is in the optimal range (approximately 15-25% of the image "
                "area contains text pixels). This means the image has a healthy amount of text "
                "relative to whitespace — not too sparse and not too crowded. OCR layout "
                "analysis will correctly identify and segment text regions."
            ),
            "good": (
                "🟡 Text density is acceptable. The image may have slightly more whitespace "
                "or slightly denser text than ideal, but OCR should handle it well. The text "
                "regions are clearly identifiable."
            ),
            "average": (
                "🟠 Text density is outside the comfortable range. The image may be mostly "
                "blank with very little text (crop tightly around the text region), or may "
                "be overcrowded with text and symbols (consider analyzing smaller sections). "
                "OCR layout analysis may struggle to properly segment columns and paragraphs."
            ),
            "poor": (
                "🔴 Extreme text density detected — either the image is nearly empty (< 3% "
                "text pixels) suggesting you uploaded a mostly blank image or the wrong region, "
                "or it is extremely dense (> 50%) suggesting the image is heavily ink-covered "
                "which may indicate binarization problems or inverted colors. Check that you "
                "have uploaded the correct image and that black text appears on a white background."
            ),
        },

        "matra_continuity_score": {
            "excellent": (
                "✅ The Devanagari Shirorekha (horizontal headline / matra) is continuous and "
                "well-preserved across the full width of text lines. This is the horizontal line "
                "that connects characters in Hindi/Sanskrit/Marathi script. A strong, unbroken "
                "matra is essential for Devanagari OCR as it defines word boundaries. "
                "The OCR engine will correctly segment words along this headline."
            ),
            "good": (
                "🟡 The Shirorekha is mostly continuous with minor breaks at a few points. "
                "Word segmentation should be mostly correct. Occasional character-level errors "
                "may occur at the break points but overall Devanagari OCR quality will be good."
            ),
            "average": (
                "🟠 Significant breaks in the Shirorekha are detected. This can happen due to "
                "document damage, fold marks, uneven ink distribution, or poor print quality. "
                "Devanagari OCR engines rely heavily on the matra for word segmentation — breaks "
                "will cause one word to be read as two or more fragments. Clean the document "
                "and ensure ink is evenly distributed before scanning."
            ),
            "poor": (
                "🔴 The Shirorekha is severely broken or nearly absent. This means the "
                "horizontal connecting line of Devanagari characters is missing in most places. "
                "This will cause catastrophic word segmentation failures in Devanagari OCR — "
                "each character cluster will be treated as a separate word. The document quality "
                "is very poor or this may not be a Devanagari text image."
            ),
        },

        "zone_integrity_score": {
            "excellent": (
                "✅ All three vertical zones of Devanagari characters are fully intact across "
                "all detected text bands. Upper zone (vowel marks above Shirorekha), middle zone "
                "(main character body), and lower zone (descenders and lower vowel marks) are all "
                "present. This means no character components are clipped or missing. OCR will "
                "correctly read all vowel signs and character modifiers."
            ),
            "good": (
                "🟡 Most text bands have all three zones intact. A small number of bands may "
                "have slightly clipped upper or lower zones. Most characters will be read "
                "correctly, with occasional vowel sign misreadings at zone boundaries."
            ),
            "average": (
                "🟠 Several text bands are missing one or more zones. This typically means the "
                "image was cropped too tightly (clipping the tops or bottoms of characters), "
                "or certain lines of text are partially obscured. Missing zones directly cause "
                "vowel sign misreadings — in Devanagari, missing a vowel sign completely "
                "changes the word's meaning."
            ),
            "poor": (
                "🔴 Most text bands have incomplete zones. Critical character components are "
                "missing from the majority of text lines. This is a serious issue — it means "
                "the image is either severely cropped, the scan missed parts of the page, "
                "or the document is physically damaged. Devanagari OCR on this image will "
                "produce mostly incorrect output with many missing vowels."
            ),
        },

        "connected_component_stability_score": {
            "excellent": (
                "✅ Character components (connected blobs of ink) are very uniform in size "
                "across the image. This indicates clean, consistent text printing with minimal "
                "noise specks or broken strokes. Each blob corresponds to one character or "
                "character part, as expected. OCR character segmentation will work perfectly."
            ),
            "good": (
                "🟡 Character components are mostly uniform with some size variation. A few "
                "extra noise specks or slightly broken strokes are present but won't cause "
                "significant segmentation problems. OCR should perform well."
            ),
            "average": (
                "🟠 Notable variation in character component sizes detected. This suggests "
                "a mix of normal characters, broken stroke fragments, and possibly noise specks "
                "all being detected as separate blobs. OCR segmentation may incorrectly group "
                "or split some characters. Apply morphological operations (dilation/erosion) "
                "to clean up broken strokes before OCR."
            ),
            "poor": (
                "🔴 Highly inconsistent character component sizes. The image likely contains "
                "many noise specks, heavily broken strokes, smudges, or mixed font sizes that "
                "create blobs ranging from tiny specks to large merged regions. OCR will "
                "struggle severely with character segmentation. Significant preprocessing "
                "(denoising, morphological cleaning) is required."
            ),
        },

        "skew_penalty_score": {
            "excellent": (
                "✅ The document is well-aligned with text lines running nearly horizontal "
                "(skew angle < 1.7°). OCR engines assume horizontal text and this image "
                "perfectly meets that assumption. Line segmentation, word spacing analysis, "
                "and reading order detection will all work correctly."
            ),
            "good": (
                "🟡 A very slight tilt is detected (approximately 1.7° - 5°). This minor "
                "skew will have minimal impact on OCR accuracy. Most OCR engines can handle "
                "this level of tilt automatically. If higher accuracy is needed, a quick "
                "deskew operation can correct it."
            ),
            "average": (
                "🟠 Noticeable skew detected (approximately 5° - 10°). Text lines are visibly "
                "tilted in the image. This causes OCR line segmentation to drift — characters "
                "at the end of long lines get assigned to the wrong line, and reading order "
                "can be disrupted. Apply a deskew algorithm (e.g., Hough transform-based) "
                "before running OCR."
            ),
            "poor": (
                "🔴 Severe skew detected (> 10°). The document is heavily rotated — text lines "
                "run at a significant angle. OCR will fail to correctly identify line boundaries "
                "and reading order. The extracted text will be garbled with characters from "
                "different lines mixed together. Rotate/deskew the image before any OCR attempt."
            ),
        },
    }

    # Select range
    if score >= 81:
        level = "excellent"
    elif score >= 61:
        level = "good"
    elif score >= 41:
        level = "average"
    else:
        level = "poor"

    factor_descs = descs.get(factor_key, {})
    return factor_descs.get(level, f"Score: {score:.1f}/100")
