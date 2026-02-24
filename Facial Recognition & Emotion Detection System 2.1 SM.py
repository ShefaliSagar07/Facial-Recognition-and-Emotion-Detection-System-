# src/__init__.py
"""Facial Recognition & Emotion Detection System"""

__version__ = "1.0.0"
__author__ = "ML Engineer"

from .face_detection import FaceDetector
from .emotion_detection import EmotionClassifier
from .utils import validate_image, save_detection_log