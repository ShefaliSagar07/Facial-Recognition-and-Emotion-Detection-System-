# src/face_detection.py
"""Face Detection Module using Multiple Approaches"""

import cv2
import numpy as np
from pathlib import Path
import streamlit as st
from typing import List, Tuple, Optional

class FaceDetector:
    """
    Face detector supporting multiple backends:
    - Haar Cascade (OpenCV)
    - MTCNN (Deep Learning)
    - Dlib
    """
    
    def __init__(self, method='haar', min_face_size=40):
        self.method = method
        self.min_face_size = min_face_size
        self.detector = self._load_detector()
        
    def _load_detector(self):
        """Load the selected face detector"""
        if self.method == 'haar':
            # Use OpenCV's Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            return cv2.CascadeClassifier(cascade_path)
        
        elif self.method == 'mtcnn':
            try:
                from facenet_pytorch import MTCNN
                return MTCNN(
                    image_size=160, 
                    margin=0,
                    min_face_size=self.min_face_size,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=True
                )
            except ImportError:
                st.warning("MTCNN not available, falling back to Haar Cascade")
                self.method = 'haar'
                return self._load_detector()
        
        elif self.method == 'dlib':
            try:
                import dlib
                return dlib.get_frontal_face_detector()
            except ImportError:
                st.warning("Dlib not available, falling back to Haar Cascade")
                self.method = 'haar'
                return self._load_detector()
        
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: BGR image (OpenCV format) or RGB
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if image is None or image.size == 0:
            return []
        
        # Convert to grayscale for Haar/Dlib
        if len(image.shape) == 3:
            if self.method == 'mtcnn':
                # MTCNN expects RGB
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        faces = []
        
        if self.method == 'haar':
            detections = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )
            faces = [(x, y, w, h) for x, y, w, h in detections]
            
        elif self.method == 'mtcnn':
            # MTCNN returns boxes in format [[x1, y1, x2, y2], ...]
            boxes, probs = self.detector.detect(gray)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    faces.append((x1, y1, x2 - x1, y2 - y1))
                    
        elif self.method == 'dlib':
            detections = self.detector(gray, 1)
            for d in detections:
                x, y, w, h = d.left(), d.top(), d.right() - d.left(), d.bottom() - d.top()
                faces.append((x, y, w, h))
        
        return faces
    
    def detect_and_crop(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect faces and return cropped face images"""
        faces = self.detect_faces(image)
        cropped_faces = []
        
        for x, y, w, h in faces:
            # Add padding
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.2)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.shape[1], x + w + pad_x)
            y2 = min(image.shape[0], y + h + pad_y)
            
            face_img = image[y1:y2, x1:x2]
            if face_img.size > 0:
                cropped_faces.append(face_img)
        
        return cropped_faces
    
    def get_face_count(self, image: np.ndarray) -> int:
        """Get number of faces in image"""
        return len(self.detect_faces(image))
    
    def draw_faces(self, image: np.ndarray, faces: List[Tuple] = None, 
                   names: List[str] = None) -> np.ndarray:
        """Draw face bounding boxes on image"""
        if faces is None:
            faces = self.detect_faces(image)
        
        result = image.copy()
        
        for idx, (x, y, w, h) in enumerate(faces):
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw label
            label = f"Face {idx + 1}"
            if names and idx < len(names):
                label = names[idx]
            
            cv2.putText(result, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result

# Singleton instance
@st.cache_resource
def get_face_detector(method='haar'):
    """Get cached face detector instance"""
    return FaceDetector(method=method)