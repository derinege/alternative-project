#!/usr/bin/env python3
"""
Study Buddy Furhat: An Engagement-Aware Conversational Robot
Main application orchestrating multimodal interaction, emotion detection, and adaptive dialogue.

Author: Derin Ege Evren
Course: ELEC 491 Senior Design Project
Advisor: Prof. Engin Erzin
"""

import os
import time
import threading
import json
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Core imports
import speech_recognition as sr
from faster_whisper import WhisperModel
import requests
import pyaudio
import wave
import io

# Web framework
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Audio processing and visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Machine Learning
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Emotion and engagement detection
from emotion_detector import EmotionDetector
from engagement_analyzer import EngagementAnalyzer
from dialogue_manager import DialogueManager
from audio_processor import AudioProcessor


class InteractionState(Enum):
    """States of user interaction"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    THINKING = "thinking"


@dataclass
class EngagementMetrics:
    """Engagement metrics for real-time monitoring"""
    timestamp: float
    voice_energy: float
    voice_emotion: str
    engagement_score: float
    attention_level: float
    response_time: float
    interaction_quality: float


@dataclass
class UserProfile:
    """User profile for personalized interaction"""
    user_id: str
    preferred_topics: List[str]
    learning_style: str
    engagement_history: List[EngagementMetrics]
    current_session_start: float
    total_interaction_time: float


class StudyBuddy:
    """
    Main Study Buddy class orchestrating all interaction components
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize Study Buddy with configuration"""
        self.config = self._load_config(config_path)
        self.state = InteractionState.IDLE
        
        # Core components
        self.audio_processor = AudioProcessor()
        self.emotion_detector = EmotionDetector()
        self.engagement_analyzer = EngagementAnalyzer()
        self.dialogue_manager = DialogueManager()
        
        # User management
        self.current_user: Optional[UserProfile] = None
        self.session_data: List[EngagementMetrics] = []
        
        # Audio processing
        self.recognizer = sr.Recognizer()
        self.whisper_model = WhisperModel(
            self.config["whisper_model"], 
            device=self.config["device"], 
            compute_type=self.config["compute_type"]
        )
        
        # Real-time processing
        self.is_listening = False
        self.processing_thread = None
        
        print("🤖 Study Buddy initialized successfully!")
        print(f"📊 Whisper model: {self.config['whisper_model']}")
        print(f"🎯 Target engagement: {self.config['target_engagement_score']}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "whisper_model": "base",
            "device": "cpu",
            "compute_type": "int8",
            "target_engagement_score": 0.7,
            "response_delay_threshold": 300,  # ms
            "emotion_detection_threshold": 0.6,
            "dialogue_model": "llama3.2:1b",
            "ollama_endpoint": "http://localhost:11434/api/generate",
            "audio_sample_rate": 16000,
            "max_interaction_duration": 3600,  # 1 hour
            "engagement_history_size": 100
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"⚠️ Config load error: {e}, using defaults")
        
        return default_config
    
    def start_session(self, user_id: str = "default_user") -> bool:
        """Start a new interaction session"""
        try:
            self.current_user = UserProfile(
                user_id=user_id,
                preferred_topics=[],
                learning_style="visual",  # Default
                engagement_history=[],
                current_session_start=time.time(),
                total_interaction_time=0.0
            )
            
            self.session_data = []
            self.state = InteractionState.IDLE
            
            print(f"🎓 Study session started for user: {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Session start error: {e}")
            return False
    
    def process_audio_input(self, audio_data: bytes) -> Dict:
        """Process incoming audio and return comprehensive analysis"""
        try:
            start_time = time.time()
            
            # Audio feature extraction
            audio_features = self.audio_processor.extract_features(audio_data)
            
            # Emotion detection from voice
            voice_emotion = self.emotion_detector.detect_emotion_from_voice(
                audio_features
            )
            
            # Speech recognition
            transcript = self._transcribe_audio(audio_data)
            
            # Engagement analysis
            engagement_score = self.engagement_analyzer.calculate_engagement(
                audio_features, voice_emotion, transcript
            )
            
            # Create metrics
            metrics = EngagementMetrics(
                timestamp=time.time(),
                voice_energy=audio_features.get('rms', 0),
                voice_emotion=voice_emotion,
                engagement_score=engagement_score,
                attention_level=self._calculate_attention_level(audio_features),
                response_time=(time.time() - start_time) * 1000,
                interaction_quality=self._assess_interaction_quality(engagement_score)
            )
            
            # Store metrics
            self.session_data.append(metrics)
            if self.current_user:
                self.current_user.engagement_history.append(metrics)
            
            return {
                'transcript': transcript,
                'emotion': voice_emotion,
                'engagement_score': engagement_score,
                'metrics': asdict(metrics),
                'processing_time': metrics.response_time
            }
            
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            return {'error': str(e)}
    
    def generate_response(self, user_input: str, context: Dict) -> Dict:
        """Generate adaptive response based on engagement and context"""
        try:
            # Get current engagement state
            current_engagement = self._get_current_engagement()
            
            # Generate response with dialogue manager
            response = self.dialogue_manager.generate_response(
                user_input=user_input,
                engagement_level=current_engagement,
                user_profile=self.current_user,
                context=context
            )
            
            # Adapt response based on engagement
            adapted_response = self._adapt_response_to_engagement(
                response, current_engagement
            )
            
            return {
                'text': adapted_response['text'],
                'tone': adapted_response['tone'],
                'feedback': adapted_response['feedback'],
                'actions': adapted_response.get('actions', []),
                'engagement_score': current_engagement
            }
            
        except Exception as e:
            print(f"❌ Response generation error: {e}")
            return {'error': str(e)}
    
    def _transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio using Whisper"""
        try:
            # Convert to WAV buffer
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.config['audio_sample_rate'])
                wf.writeframes(audio_data)
            wav_buffer.seek(0)
            
            # Transcribe with Whisper
            segments, _ = self.whisper_model.transcribe(
                wav_buffer,
                beam_size=1,
                temperature=0.0,
                language=None,  # Auto-detect language (Turkish/English)
                condition_on_previous_text=False
            )
            
            # Combine segments
            text = " ".join([segment.text for segment in segments])
            return text.strip()
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
    
    def _calculate_attention_level(self, audio_features: Dict) -> float:
        """Calculate attention level from audio features"""
        # Simple heuristic based on voice energy and consistency
        energy = audio_features.get('rms', 0)
        consistency = audio_features.get('dynamic_range', 1)
        
        # Normalize and combine features
        attention = min(1.0, energy / 1000.0) * consistency
        return max(0.0, min(1.0, attention))
    
    def _assess_interaction_quality(self, engagement_score: float) -> float:
        """Assess overall interaction quality"""
        # Consider recent engagement trend
        if len(self.session_data) >= 3:
            recent_engagement = [m.engagement_score for m in self.session_data[-3:]]
            trend = np.mean(np.diff(recent_engagement))
            quality = engagement_score + (trend * 0.2)
        else:
            quality = engagement_score
        
        return max(0.0, min(1.0, quality))
    
    def _get_current_engagement(self) -> float:
        """Get current engagement level"""
        if not self.session_data:
            return 0.5  # Neutral
        
        # Use weighted average of recent interactions
        recent_metrics = self.session_data[-5:]
        weights = np.linspace(0.5, 1.0, len(recent_metrics))
        weighted_engagement = np.average(
            [m.engagement_score for m in recent_metrics],
            weights=weights
        )
        
        return weighted_engagement
    
    def _adapt_response_to_engagement(self, response: Dict, engagement: float) -> Dict:
        """Adapt response based on current engagement level"""
        if engagement < 0.3:
            # Low engagement - try to re-engage
            response['tone'] = 'encouraging'
            response['feedback'] = "I notice you might be losing focus. Let's try something different!"
            response['actions'] = ['change_topic', 'use_visuals', 'take_break']
        elif engagement > 0.8:
            # High engagement - maintain momentum
            response['tone'] = 'enthusiastic'
            response['feedback'] = "Great! You're really engaged. Let's keep this momentum going!"
            response['actions'] = ['continue_topic', 'increase_difficulty']
        else:
            # Medium engagement - normal interaction
            response['tone'] = 'neutral'
            response['feedback'] = "Good progress. Let's continue."
        
        return response
    
    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary"""
        if not self.session_data:
            return {'error': 'No session data available'}
        
        metrics = [asdict(m) for m in self.session_data]
        
        return {
            'session_duration': time.time() - self.current_user.current_session_start if self.current_user else 0,
            'total_interactions': len(self.session_data),
            'average_engagement': np.mean([m['engagement_score'] for m in metrics]),
            'engagement_trend': self._calculate_engagement_trend(),
            'dominant_emotions': self._get_dominant_emotions(),
            'recommendations': self._generate_recommendations(),
            'detailed_metrics': metrics
        }
    
    def _calculate_engagement_trend(self) -> str:
        """Calculate engagement trend over session"""
        if len(self.session_data) < 3:
            return "insufficient_data"
        
        recent_engagement = [m.engagement_score for m in self.session_data[-5:]]
        early_engagement = [m.engagement_score for m in self.session_data[:5]]
        
        recent_avg = np.mean(recent_engagement)
        early_avg = np.mean(early_engagement)
        
        if recent_avg > early_avg + 0.1:
            return "improving"
        elif recent_avg < early_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _get_dominant_emotions(self) -> List[str]:
        """Get most frequent emotions in session"""
        if not self.session_data:
            return []
        
        emotions = [m.voice_emotion for m in self.session_data]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Return top 3 emotions
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        return [emotion for emotion, count in sorted_emotions[:3]]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate study recommendations based on session data"""
        recommendations = []
        
        if not self.session_data:
            return ["Start with a simple topic to build engagement"]
        
        avg_engagement = np.mean([m.engagement_score for m in self.session_data])
        
        if avg_engagement < 0.4:
            recommendations.extend([
                "Consider taking a short break",
                "Try interactive learning methods",
                "Use visual aids or examples"
            ])
        elif avg_engagement > 0.8:
            recommendations.extend([
                "Great engagement! Continue with current approach",
                "Consider more challenging topics",
                "Maintain this learning pace"
            ])
        else:
            recommendations.extend([
                "Good progress, consider varying activities",
                "Monitor attention levels",
                "Continue current study method"
            ])
        
        return recommendations


def main():
    """Main function for testing Study Buddy"""
    print("🎓 Study Buddy Furhat - Engagement-Aware Conversational Robot")
    print("=" * 70)
    
    # Initialize Study Buddy
    buddy = StudyBuddy()
    
    # Start a test session
    if buddy.start_session("test_user"):
        print("✅ Test session started successfully")
        print("🤖 Study Buddy is ready for interaction!")
        print("💡 Use the web interface to interact with Study Buddy")
    else:
        print("❌ Failed to start test session")


if __name__ == "__main__":
    main()
