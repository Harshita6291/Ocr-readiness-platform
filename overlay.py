"""
Visual Diagnostics Overlay Generator for OCR Readiness Platform.
Generates explainable visual maps for Blur, Noise, Skew, Text Density, Stroke Width, and Connected Components.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple


def get_blur_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """
    Generates a heatmap of sharp (red/yellow) vs blurry (blue/cyan) regions using local Laplacian variance.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Compute local Laplacian magnitude
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    abs_lap = np.abs(lap)
    
    # Smooth local sharpness map
    kernel_size = max(15, min(h, w) // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    sharpness_map = cv2.GaussianBlur(abs_lap, (kernel_size, kernel_size), 0)
    
    # Normalize map to 0-255
    norm_map = cv2.normalize(sharpness_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply JET color map (Red=High Sharpness, Blue=Blurry)
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    
    # Blend with original image
    overlay = cv2.addWeighted(img_bgr, 0.45, heatmap, 0.55, 0)
    return overlay


def get_noise_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """
    Highlights noisy background patches in red/yellow while ignoring text edges.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Exclude text stroke edges
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((7, 7), np.uint8))
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = np.abs(gray - blurred)
    
    # Zero out edge regions so only flat noise is mapped
    residual[edges_dilated > 0] = 0
    
    # Normalize noise intensity map
    norm_noise = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Color map for noise: Hot colormap (black=clean, red/yellow=noisy)
    noise_colored = cv2.applyColorMap(norm_noise, cv2.COLORMAP_HOT)
    
    overlay = cv2.addWeighted(img_bgr, 0.6, noise_colored, 0.4, 0)
    return overlay


def get_skew_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """
    Draws horizontal baseline (cyan dashed) and detected text skew angle line (red/green) with degree annotation.
    """
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return canvas
        
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = -(90 + angle)
    skew_deg = abs(angle)
    if skew_deg > 45:
        skew_deg = 90 - skew_deg
        
    center = (w // 2, h // 2)
    length = int(min(w, h) * 0.4)
    
    # True horizontal reference line (cyan)
    cv2.line(canvas, (center[0] - length, center[1]), (center[0] + length, center[1]), (255, 255, 0), 2, cv2.LINE_AA)
    
    # Angle vector line (Red if high skew, Green if low skew)
    rad = np.radians(angle)
    dx = int(length * np.cos(rad))
    dy = int(length * np.sin(rad))
    line_color = (0, 255, 0) if skew_deg < 3 else (0, 0, 255)
    
    cv2.line(canvas, (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy), line_color, 3, cv2.LINE_AA)
    
    # Add text banner
    banner_txt = f"Detected Skew: {angle:.2f} degrees"
    cv2.rectangle(canvas, (10, 10), (360, 50), (26, 43, 74), -1)
    cv2.putText(canvas, banner_txt, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    return canvas


def get_text_density_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """
    Draws bounding boxes around detected text regions and blocks.
    """
    canvas = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological dilation to merge nearby text characters into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 8:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 196, 180), 2)
            
    return canvas


def get_stroke_width_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """
    Color-codes stroke pixels: ideal (1.5-4px = Green), thin (<1.5px = Red), thick (>4px = Orange).
    """
    canvas = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    diameter = dist * 2.0  # stroke width approximation
    
    # Create colored stroke map
    stroke_colored = canvas.copy()
    
    # Thin strokes (< 1.5px) -> Red
    thin_mask = (diameter > 0) & (diameter < 1.5)
    stroke_colored[thin_mask] = [0, 0, 255]
    
    # Ideal strokes (1.5 - 4.5px) -> Green
    ideal_mask = (diameter >= 1.5) & (diameter <= 4.5)
    stroke_colored[ideal_mask] = [0, 255, 0]
    
    # Thick strokes (> 4.5px) -> Orange
    thick_mask = (diameter > 4.5)
    stroke_colored[thick_mask] = [0, 140, 255]
    
    # Blend over original text
    text_pixels = binary > 0
    canvas[text_pixels] = cv2.addWeighted(canvas[text_pixels], 0.3, stroke_colored[text_pixels], 0.7, 0)
    
    return canvas


def get_connected_components_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """
    Draws bounding boxes around each detected connected component.
    """
    canvas = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Generate pseudo-random colors for components
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_labels, 3), dtype=np.uint8)
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= 4:  # filter noise specks
            color = [int(c) for c in colors[i]]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
            
    return canvas


OVERLAY_FUNCTIONS = {
    "Blur Heatmap": get_blur_overlay,
    "Noise Map": get_noise_overlay,
    "Skew Angle Line": get_skew_overlay,
    "Text Density BBoxes": get_text_density_overlay,
    "Stroke Width Map": get_stroke_width_overlay,
    "Connected Components": get_connected_components_overlay,
}
