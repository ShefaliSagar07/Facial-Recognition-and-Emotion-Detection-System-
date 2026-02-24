# src/emotion_detection.py
"""Emotion Detection using Deep Learning"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import streamlit as st
from typing import List, Dict, Tuple
import pickle

class EmotionClassifier:
    """
    Emotion classifier using CNN or Transfer Learning.
    Supports 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
    """
    
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self, model_path: str = None, use_pretrained=True):
        self.model = None
        self.model_path = model_path or "models/emotion_model.h5"
        self.img_size = (48, 48)  # Standard for emotion models
        self.use_pretrained = use_pretrained
        self._load_model()
        
    def _load_model(self):
        """Load or create emotion classification model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                st.success("Emotion model loaded successfully!")
            else:
                if self.use_pretrained:
                    st.warning(f"Model not found at {self.model_path}. Using simple classifier.")
                self.model = self._create_simple_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.model = self._create_simple_model()
    
    def _create_simple_model(self):
        """Create a simple CNN for demonstration"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for emotion classification"""
        # Resize to target size
        img = cv2.resize(face_img, self.img_size)
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRGRAY)
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        img = np.expand_dims(img, axis=-1)