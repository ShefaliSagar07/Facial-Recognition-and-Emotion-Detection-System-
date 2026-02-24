# src/utils.py
"""Utility functions for the Facial Emotion System"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Create necessary directories
UPLOAD_DIR = Path("uploads")
LOG_DIR = Path("logs")
MODEL_DIR = Path("models")

for dir_path in [UPLOAD_DIR, LOG_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def get_file_hash(file_path):
    """Generate hash for file to avoid duplicates"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def validate_image_file(uploaded_file):
    """Validate uploaded image file"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if uploaded_file.type not in allowed_types:
        return False, f"Invalid file type. Allowed: {allowed_types}"
    
    # Check file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "File too large. Max size is 10MB"
    
    return True, "Valid"

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return path"""
    file_ext = uploaded_file.name.split('.')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    filepath = UPLOAD_DIR / filename
    
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return filepath

def save_detection_log(results, image_path=None):
    """Save detection results to log file"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "image_path": str(image_path) if image_path else None,
        "results": results
    }
    
    log_file = LOG_DIR / "detections.json"
    
    # Read existing logs
    logs = []
    if log_file.exists():
        with open(log_file, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    
    logs.append(log_entry)
    
    # Keep only last 1000 entries
    logs = logs[-1000:]
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

def load_detection_logs(limit=100):
    """Load recent detection logs"""
    log_file = LOG_DIR / "detections.json"
    
    if not log_file.exists():
        return []
    
    with open(log_file, 'r') as f:
        try:
            logs = json.load(f)
            return logs[-limit:]
        except json.JSONDecodeError:
            return []

def pil_to_cv2(image):
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image):
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def draw_faces_boxes(image, faces, names=None, emotions=None):
    """Draw bounding boxes and labels on image"""
    img_copy = image.copy()
    
    for idx, face in enumerate(faces):
        x, y, w, h = face
        
        # Determine box color based on emotion
        box_color = (0, 255, 0)  # Green default
        
        if emotions and idx < len(emotions):
            emotion = emotions[idx].lower()
            if 'angry' in emotion:
                box_color = (0, 0, 255)  # Red
            elif 'happy' in emotion or 'joy' in emotion:
                box_color = (0, 255, 0)  # Green
            elif 'sad' in emotion:
                box_color = (255, 0, 0)  # Blue
            elif 'surprise' in emotion:
                box_color = (255, 255, 0)  # Yellow
            else:
                box_color = (255, 165, 0)  # Orange
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), box_color, 2)
        
        # Draw label
        label = f"Face {idx + 1}"
        if names and idx < len(names):
            label = f"{names[idx]}"
        if emotions and idx < len(emotions):
            label += f" - {emotions[idx]}"
        
        cv2.putText(img_copy, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    return img_copy

def get_emotion_color(emotion):
    """Get color for emotion"""
    colors = {
        'angry': '#e74c3c',
        'disgust': '#9b59b6',
        'fear': '#3498db',
        'happy': '#2ecc71',
        'sad': '#1abc9c',
        'surprise': '#f1c40f',
        'neutral': '#95a5a6'
    }
    return colors.get(emotion.lower(), '#ffffff')

def create_emoji_for_emotion(emotion):
    """Get emoji representation of emotion"""
    emojis = {
        'angry': '😠',
        'disgust': '🤢',
        'fear': '😨',
        'happy': '😊',
        'sad': '😢',
        'surprise': '😲',
        'neutral': '😐'
    }
    return emojis.get(emotion.lower(), '😐')