#!/usr/bin/env python3
"""
Emotion Detection Module for Study Buddy
Detects emotions from voice features and facial expressions using multimodal analysis.

Author: Derin Ege Evren
Course: ELEC 491 Senior Design Project
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import cv2
from dataclasses import dataclass


@dataclass
class EmotionFeatures:
    """Voice features for emotion detection"""
    pitch: float
    energy: float
    spectral_centroid: float
    mfcc_1: float
    mfcc_2: float
    mfcc_3: float
    zero_crossing_rate: float
    spectral_rolloff: float
    tempo: float


class VoiceEmotionDetector:
    """Voice-based emotion detection using audio features"""
    
    def __init__(self, model_path: str = None):
        """Initialize voice emotion detector"""
        self.scaler = StandardScaler()
        self.model = None
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize with a simple heuristic-based model"""
        print("🎭 Initializing default emotion detection model...")
        
        # Simple rule-based emotion detection as fallback
        self.model = None
        self.feature_weights = {
            'pitch': 0.3,
            'energy': 0.25,
            'spectral_centroid': 0.2,
            'zero_crossing_rate': 0.15,
            'tempo': 0.1
        }
    
    def extract_voice_features(self, audio_features: Dict) -> EmotionFeatures:
        """Extract emotion-relevant features from audio"""
        try:
            # Extract basic audio features
            energy = audio_features.get('rms', 0)
            spectral_centroid = audio_features.get('dominant_freq', 0)
            
            # Calculate pitch (simplified)
            pitch = self._estimate_pitch(audio_features)
            
            # Calculate MFCC-like features (simplified)
            mfcc_1 = energy * 0.8
            mfcc_2 = spectral_centroid * 0.6
            mfcc_3 = energy * spectral_centroid * 0.4
            
            # Zero crossing rate (simplified)
            zcr = audio_features.get('zero_crossing_rate', 0.1)
            
            # Spectral rolloff (simplified)
            rolloff = spectral_centroid * 1.2
            
            # Tempo estimation (simplified)
            tempo = self._estimate_tempo(audio_features)
            
            return EmotionFeatures(
                pitch=pitch,
                energy=energy,
                spectral_centroid=spectral_centroid,
                mfcc_1=mfcc_1,
                mfcc_2=mfcc_2,
                mfcc_3=mfcc_3,
                zero_crossing_rate=zcr,
                spectral_rolloff=rolloff,
                tempo=tempo
            )
            
        except Exception as e:
            print(f"❌ Voice feature extraction error: {e}")
            return EmotionFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _estimate_pitch(self, audio_features: Dict) -> float:
        """Estimate pitch from audio features"""
        # Simplified pitch estimation
        dominant_freq = audio_features.get('dominant_freq', 0)
        energy = audio_features.get('rms', 0)
        
        # Basic pitch estimation (Hz to normalized pitch)
        if dominant_freq > 0:
            # Normalize to typical human voice range (80-300 Hz)
            normalized_pitch = (dominant_freq - 80) / 220
            return max(0, min(1, normalized_pitch))
        return 0.5  # Neutral pitch
    
    def _estimate_tempo(self, audio_features: Dict) -> float:
        """Estimate speech tempo from audio features"""
        # Simplified tempo estimation based on energy variations
        energy = audio_features.get('rms', 0)
        dynamic_range = audio_features.get('dynamic_range', 1)
        
        # Higher energy and dynamic range suggest faster speech
        tempo = (energy / 1000.0) * dynamic_range
        return min(1.0, tempo)
    
    def detect_emotion(self, features: EmotionFeatures) -> Tuple[str, float]:
        """Detect emotion from voice features"""
        try:
            if self.model:
                # Use trained model
                feature_vector = np.array([
                    features.pitch, features.energy, features.spectral_centroid,
                    features.mfcc_1, features.mfcc_2, features.mfcc_3,
                    features.zero_crossing_rate, features.spectral_rolloff, features.tempo
                ]).reshape(1, -1)
                
                feature_vector = self.scaler.transform(feature_vector)
                emotion_probs = self.model.predict_proba(feature_vector)[0]
                emotion_idx = np.argmax(emotion_probs)
                confidence = emotion_probs[emotion_idx]
                
                return self.emotion_labels[emotion_idx], confidence
            
            else:
                # Use heuristic-based detection
                return self._heuristic_emotion_detection(features)
                
        except Exception as e:
            print(f"❌ Emotion detection error: {e}")
            return "neutral", 0.5
    
    def _heuristic_emotion_detection(self, features: EmotionFeatures) -> Tuple[str, float]:
        """Heuristic-based emotion detection"""
        # Simple rule-based emotion detection
        
        # Happy: High pitch, high energy, moderate tempo
        if features.pitch > 0.6 and features.energy > 0.5 and 0.4 < features.tempo < 0.8:
            return "happy", 0.7
        
        # Sad: Low pitch, low energy, slow tempo
        elif features.pitch < 0.4 and features.energy < 0.3 and features.tempo < 0.4:
            return "sad", 0.7
        
        # Angry: High energy, high tempo, variable pitch
        elif features.energy > 0.7 and features.tempo > 0.7 and features.zero_crossing_rate > 0.5:
            return "angry", 0.7
        
        # Surprised: High pitch, high energy, high tempo
        elif features.pitch > 0.7 and features.energy > 0.6 and features.tempo > 0.6:
            return "surprised", 0.7
        
        # Fearful: High pitch, low energy, variable tempo
        elif features.pitch > 0.6 and features.energy < 0.4:
            return "fearful", 0.6
        
        # Neutral: Balanced features
        else:
            return "neutral", 0.5
    
    def load_model(self, model_path: str):
        """Load pre-trained emotion detection model"""
        try:
            self.model = joblib.load(f"{model_path}_model.pkl")
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            print(f"✅ Loaded emotion detection model from {model_path}")
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}, using heuristic detection")


class FacialEmotionDetector:
    """Facial expression-based emotion detection"""
    
    def __init__(self):
        """Initialize facial emotion detector"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        
        # Simple emotion detection based on facial geometry
        self.emotion_thresholds = {
            'happy': {'mouth_ratio': 1.2, 'eye_ratio': 1.1},
            'sad': {'mouth_ratio': 0.8, 'eye_ratio': 0.9},
            'surprised': {'eye_ratio': 1.3, 'eyebrow_ratio': 1.2},
            'angry': {'eyebrow_ratio': 0.8, 'mouth_ratio': 0.7}
        }
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in video frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces.tolist()
    
    def analyze_facial_emotion(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """Analyze facial emotion from face region"""
        try:
            x, y, w, h = face_bbox
            face_roi = frame[y:y+h, x:x+w]
            
            # Simple facial feature analysis
            emotion, confidence = self._analyze_facial_features(face_roi)
            
            return emotion, confidence
            
        except Exception as e:
            print(f"❌ Facial emotion analysis error: {e}")
            return "neutral", 0.5
    
    def _analyze_facial_features(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Analyze facial features for emotion detection"""
        # Simplified facial emotion detection
        # In a full implementation, this would use more sophisticated computer vision
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Simple feature extraction (placeholder)
        # In practice, you'd use more sophisticated methods like:
        # - Facial landmark detection
        # - Deep learning models (FER2013, AffectNet)
        # - Optical flow analysis
        
        # For now, return neutral emotion
        return "neutral", 0.5


class EmotionDetector:
    """Main emotion detection class combining voice and facial analysis"""
    
    def __init__(self, voice_model_path: str = None):
        """Initialize multimodal emotion detector"""
        self.voice_detector = VoiceEmotionDetector(voice_model_path)
        self.facial_detector = FacialEmotionDetector()
        
        # Fusion weights
        self.voice_weight = 0.7  # Voice is primary for Study Buddy
        self.facial_weight = 0.3
        
        print("🎭 Multimodal emotion detector initialized")
    
    def detect_emotion_from_voice(self, audio_features: Dict) -> str:
        """Detect emotion from voice features only"""
        try:
            features = self.voice_detector.extract_voice_features(audio_features)
            emotion, confidence = self.voice_detector.detect_emotion(features)
            
            # Log emotion detection
            print(f"🎭 Voice emotion: {emotion} (confidence: {confidence:.2f})")
            
            return emotion
            
        except Exception as e:
            print(f"❌ Voice emotion detection error: {e}")
            return "neutral"
    
    def detect_emotion_from_face(self, frame: np.ndarray) -> str:
        """Detect emotion from facial expression only"""
        try:
            faces = self.facial_detector.detect_faces(frame)
            
            if not faces:
                return "neutral"
            
            # Analyze the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            emotion, confidence = self.facial_detector.analyze_facial_emotion(frame, largest_face)
            
            print(f"🎭 Facial emotion: {emotion} (confidence: {confidence:.2f})")
            
            return emotion
            
        except Exception as e:
            print(f"❌ Facial emotion detection error: {e}")
            return "neutral"
    
    def detect_multimodal_emotion(self, audio_features: Dict, video_frame: np.ndarray = None) -> Tuple[str, float]:
        """Detect emotion using both voice and facial features"""
        try:
            # Voice emotion
            voice_emotion = self.detect_emotion_from_voice(audio_features)
            voice_confidence = 0.7  # Placeholder
            
            # Facial emotion (if video available)
            if video_frame is not None:
                facial_emotion = self.detect_emotion_from_face(video_frame)
                facial_confidence = 0.6  # Placeholder
            else:
                facial_emotion = "neutral"
                facial_confidence = 0.0
            
            # Fuse emotions
            if facial_emotion == "neutral" or video_frame is None:
                # Voice only
                return voice_emotion, voice_confidence
            else:
                # Weighted fusion
                if voice_emotion == facial_emotion:
                    # Agreement - higher confidence
                    return voice_emotion, (voice_confidence + facial_confidence) / 2
                else:
                    # Disagreement - prefer voice
                    return voice_emotion, voice_confidence * self.voice_weight
            
        except Exception as e:
            print(f"❌ Multimodal emotion detection error: {e}")
            return "neutral", 0.5
    
    def get_emotion_intensity(self, emotion: str, confidence: float) -> str:
        """Get emotion intensity level"""
        if confidence < 0.3:
            return "weak"
        elif confidence < 0.6:
            return "moderate"
        else:
            return "strong"
    
    def is_positive_emotion(self, emotion: str) -> bool:
        """Check if emotion is positive for learning"""
        positive_emotions = ['happy', 'surprised', 'neutral']
        return emotion in positive_emotions
    
    def is_negative_emotion(self, emotion: str) -> bool:
        """Check if emotion might indicate learning difficulty"""
        negative_emotions = ['sad', 'angry', 'fearful', 'disgusted']
        return emotion in negative_emotions


def main():
    """Test emotion detection module"""
    print("🎭 Emotion Detection Module Test")
    print("=" * 40)
    
    # Initialize emotion detector
    detector = EmotionDetector()
    
    # Test with sample audio features
    sample_audio_features = {
        'rms': 500,
        'dominant_freq': 150,
        'dynamic_range': 2.5,
        'zero_crossing_rate': 0.3
    }
    
    # Test voice emotion detection
    emotion = detector.detect_emotion_from_voice(sample_audio_features)
    print(f"🎤 Detected emotion: {emotion}")
    
    # Test emotion classification
    print(f"😊 Positive emotion: {detector.is_positive_emotion(emotion)}")
    print(f"😞 Negative emotion: {detector.is_negative_emotion(emotion)}")


if __name__ == "__main__":
    import os
    main()
