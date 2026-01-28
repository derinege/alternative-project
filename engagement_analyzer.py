#!/usr/bin/env python3
"""
Engagement Analysis Module for Study Buddy
Analyzes user engagement through multimodal signals and interaction patterns.

Author: Derin Ege Evren
Course: ELEC 491 Senior Design Project
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import math


@dataclass
class EngagementSignal:
    """Individual engagement signal measurement"""
    timestamp: float
    signal_type: str  # 'voice', 'facial', 'behavioral'
    value: float
    confidence: float


@dataclass
class EngagementWindow:
    """Engagement analysis window"""
    start_time: float
    end_time: float
    signals: List[EngagementSignal]
    overall_score: float
    trend: str  # 'increasing', 'decreasing', 'stable'


class EngagementAnalyzer:
    """
    Analyzes user engagement through multiple modalities and temporal patterns
    """
    
    def __init__(self, window_size: int = 10, history_size: int = 100):
        """Initialize engagement analyzer"""
        self.window_size = window_size  # Number of recent signals to analyze
        self.history_size = history_size  # Maximum history to keep
        
        # Signal buffers
        self.voice_signals = deque(maxlen=history_size)
        self.facial_signals = deque(maxlen=history_size)
        self.behavioral_signals = deque(maxlen=history_size)
        
        # Engagement windows
        self.engagement_windows = deque(maxlen=50)
        
        # Weights for different signal types
        self.signal_weights = {
            'voice': 0.4,
            'facial': 0.3,
            'behavioral': 0.3
        }
        
        # Engagement thresholds
        self.thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
        print("📊 Engagement analyzer initialized")
        print(f"📈 Window size: {window_size}, History size: {history_size}")
    
    def add_voice_signal(self, energy: float, emotion: str, confidence: float = 0.8):
        """Add voice-based engagement signal"""
        signal_value = self._calculate_voice_engagement(energy, emotion)
        
        signal = EngagementSignal(
            timestamp=time.time(),
            signal_type='voice',
            value=signal_value,
            confidence=confidence
        )
        
        self.voice_signals.append(signal)
        print(f"🎤 Voice signal: {signal_value:.3f} (emotion: {emotion})")
    
    def add_facial_signal(self, attention_score: float, emotion: str, confidence: float = 0.7):
        """Add facial-based engagement signal"""
        signal_value = self._calculate_facial_engagement(attention_score, emotion)
        
        signal = EngagementSignal(
            timestamp=time.time(),
            signal_type='facial',
            value=signal_value,
            confidence=confidence
        )
        
        self.facial_signals.append(signal)
        print(f"👁️ Facial signal: {signal_value:.3f} (attention: {attention_score:.3f})")
    
    def add_behavioral_signal(self, response_time: float, interaction_quality: float, confidence: float = 0.9):
        """Add behavioral engagement signal"""
        signal_value = self._calculate_behavioral_engagement(response_time, interaction_quality)
        
        signal = EngagementSignal(
            timestamp=time.time(),
            signal_type='behavioral',
            value=signal_value,
            confidence=confidence
        )
        
        self.behavioral_signals.append(signal)
        print(f"🎯 Behavioral signal: {signal_value:.3f} (RT: {response_time:.1f}ms)")
    
    def calculate_engagement(self, audio_features: Dict, emotion: str, transcript: str) -> float:
        """Calculate overall engagement score from current inputs"""
        try:
            current_time = time.time()
            
            # Voice engagement
            voice_engagement = self._calculate_voice_engagement(
                audio_features.get('rms', 0),
                emotion
            )
            
            # Behavioral engagement from transcript
            behavioral_engagement = self._calculate_text_engagement(transcript)
            
            # Temporal engagement (based on recent patterns)
            temporal_engagement = self._calculate_temporal_engagement()
            
            # Combine signals
            weights = [0.5, 0.3, 0.2]  # voice, behavioral, temporal
            engagements = [voice_engagement, behavioral_engagement, temporal_engagement]
            
            overall_engagement = np.average(engagements, weights=weights)
            
            # Add signals to buffers
            self.add_voice_signal(audio_features.get('rms', 0), emotion)
            self.add_behavioral_signal(0, behavioral_engagement)  # Response time unknown here
            
            # Update engagement windows
            self._update_engagement_windows(current_time, overall_engagement)
            
            print(f"📊 Overall engagement: {overall_engagement:.3f}")
            return max(0.0, min(1.0, overall_engagement))
            
        except Exception as e:
            print(f"❌ Engagement calculation error: {e}")
            return 0.5  # Neutral engagement
    
    def _calculate_voice_engagement(self, energy: float, emotion: str) -> float:
        """Calculate engagement from voice characteristics"""
        # Base engagement from energy level
        energy_score = min(1.0, energy / 1000.0)
        
        # Emotion-based adjustment
        emotion_multipliers = {
            'happy': 1.2,
            'surprised': 1.1,
            'neutral': 1.0,
            'sad': 0.7,
            'angry': 0.8,
            'fearful': 0.6,
            'disgusted': 0.5
        }
        
        emotion_multiplier = emotion_multipliers.get(emotion, 1.0)
        voice_engagement = energy_score * emotion_multiplier
        
        return max(0.0, min(1.0, voice_engagement))
    
    def _calculate_facial_engagement(self, attention_score: float, emotion: str) -> float:
        """Calculate engagement from facial features"""
        # Base engagement from attention
        attention_engagement = attention_score
        
        # Emotion adjustment
        emotion_adjustments = {
            'happy': 0.2,
            'surprised': 0.1,
            'neutral': 0.0,
            'sad': -0.2,
            'angry': -0.1,
            'fearful': -0.3,
            'disgusted': -0.2
        }
        
        emotion_adjustment = emotion_adjustments.get(emotion, 0.0)
        facial_engagement = attention_engagement + emotion_adjustment
        
        return max(0.0, min(1.0, facial_engagement))
    
    def _calculate_behavioral_engagement(self, response_time: float, interaction_quality: float) -> float:
        """Calculate engagement from behavioral patterns"""
        # Response time component (faster responses = higher engagement)
        # Optimal response time: 1-3 seconds
        if 1000 <= response_time <= 3000:
            time_score = 1.0
        elif response_time < 1000:
            time_score = 0.8  # Too fast might indicate not thinking
        else:
            # Slower responses get lower scores
            time_score = max(0.2, 1.0 - (response_time - 3000) / 10000)
        
        # Interaction quality component
        quality_score = interaction_quality
        
        # Combine time and quality
        behavioral_engagement = (time_score * 0.6) + (quality_score * 0.4)
        
        return max(0.0, min(1.0, behavioral_engagement))
    
    def _calculate_text_engagement(self, transcript: str) -> float:
        """Calculate engagement from text content"""
        if not transcript or len(transcript.strip()) < 3:
            return 0.3  # Low engagement for very short responses
        
        # Length component (moderate length = good engagement)
        length = len(transcript.split())
        if 3 <= length <= 15:
            length_score = 1.0
        elif length < 3:
            length_score = 0.5
        else:
            length_score = max(0.6, 1.0 - (length - 15) / 20)
        
        # Question indicators (asking questions = high engagement)
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who']
        has_questions = any(indicator in transcript.lower() for indicator in question_indicators)
        question_score = 1.2 if has_questions else 1.0
        
        # Engagement words
        engagement_words = ['yes', 'sure', 'okay', 'great', 'interesting', 'help', 'understand']
        has_engagement_words = any(word in transcript.lower() for word in engagement_words)
        engagement_word_score = 1.1 if has_engagement_words else 1.0
        
        # Disengagement words
        disengagement_words = ['no', 'don\'t', 'can\'t', 'won\'t', 'boring', 'tired', 'stop']
        has_disengagement_words = any(word in transcript.lower() for word in disengagement_words)
        disengagement_word_score = 0.7 if has_disengagement_words else 1.0
        
        text_engagement = length_score * question_score * engagement_word_score * disengagement_word_score
        
        return max(0.0, min(1.0, text_engagement))
    
    def _calculate_temporal_engagement(self) -> float:
        """Calculate engagement based on temporal patterns"""
        if len(self.voice_signals) < 3:
            return 0.5  # Neutral if insufficient data
        
        # Recent trend analysis
        recent_signals = list(self.voice_signals)[-self.window_size:]
        if len(recent_signals) >= 3:
            values = [s.value for s in recent_signals]
            
            # Calculate trend
            if len(values) >= 3:
                early_avg = np.mean(values[:len(values)//2])
                late_avg = np.mean(values[len(values)//2:])
                trend = late_avg - early_avg
                
                # Convert trend to engagement score
                if trend > 0.1:
                    return 0.8  # Increasing engagement
                elif trend < -0.1:
                    return 0.3  # Decreasing engagement
                else:
                    return 0.6  # Stable engagement
        
        return 0.5  # Default neutral
    
    def _update_engagement_windows(self, current_time: float, engagement_score: float):
        """Update engagement analysis windows"""
        # Create new window if enough time has passed
        if not self.engagement_windows or current_time - self.engagement_windows[-1].end_time > 30:
            # Get signals from last 30 seconds
            recent_signals = []
            for signal_list in [self.voice_signals, self.facial_signals, self.behavioral_signals]:
                recent_signals.extend([
                    s for s in signal_list 
                    if current_time - s.timestamp <= 30
                ])
            
            # Calculate window metrics
            window_score = engagement_score
            trend = self._calculate_window_trend(recent_signals)
            
            window = EngagementWindow(
                start_time=current_time - 30,
                end_time=current_time,
                signals=recent_signals,
                overall_score=window_score,
                trend=trend
            )
            
            self.engagement_windows.append(window)
    
    def _calculate_window_trend(self, signals: List[EngagementSignal]) -> str:
        """Calculate trend for a window of signals"""
        if len(signals) < 4:
            return "stable"
        
        # Sort by timestamp
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)
        values = [s.value for s in sorted_signals]
        
        # Calculate trend
        early_avg = np.mean(values[:len(values)//2])
        late_avg = np.mean(values[len(values)//2:])
        
        if late_avg > early_avg + 0.1:
            return "increasing"
        elif late_avg < early_avg - 0.1:
            return "decreasing"
        else:
            return "stable"
    
    def get_engagement_level(self, score: float) -> str:
        """Get engagement level category"""
        if score < self.thresholds['low']:
            return "low"
        elif score < self.thresholds['medium']:
            return "medium"
        elif score < self.thresholds['high']:
            return "high"
        else:
            return "very_high"
    
    def get_engagement_trend(self) -> str:
        """Get overall engagement trend"""
        if len(self.engagement_windows) < 2:
            return "insufficient_data"
        
        recent_windows = list(self.engagement_windows)[-3:]
        trends = [w.trend for w in recent_windows]
        
        # Majority trend
        if trends.count('increasing') > len(trends) // 2:
            return "improving"
        elif trends.count('decreasing') > len(trends) // 2:
            return "declining"
        else:
            return "stable"
    
    def get_engagement_summary(self) -> Dict:
        """Get comprehensive engagement summary"""
        if not self.voice_signals:
            return {'error': 'No engagement data available'}
        
        # Recent engagement
        recent_voice = list(self.voice_signals)[-10:] if self.voice_signals else []
        recent_facial = list(self.facial_signals)[-10:] if self.facial_signals else []
        recent_behavioral = list(self.behavioral_signals)[-10:] if self.behavioral_signals else []
        
        # Calculate averages
        avg_voice = np.mean([s.value for s in recent_voice]) if recent_voice else 0.5
        avg_facial = np.mean([s.value for s in recent_facial]) if recent_facial else 0.5
        avg_behavioral = np.mean([s.value for s in recent_behavioral]) if recent_behavioral else 0.5
        
        # Overall average
        overall_avg = (avg_voice * self.signal_weights['voice'] + 
                      avg_facial * self.signal_weights['facial'] + 
                      avg_behavioral * self.signal_weights['behavioral'])
        
        return {
            'overall_engagement': overall_avg,
            'engagement_level': self.get_engagement_level(overall_avg),
            'voice_engagement': avg_voice,
            'facial_engagement': avg_facial,
            'behavioral_engagement': avg_behavioral,
            'trend': self.get_engagement_trend(),
            'total_signals': len(self.voice_signals) + len(self.facial_signals) + len(self.behavioral_signals),
            'recent_windows': len(self.engagement_windows)
        }
    
    def should_intervene(self, engagement_score: float) -> Tuple[bool, str]:
        """Determine if intervention is needed"""
        if engagement_score < self.thresholds['low']:
            return True, "low_engagement"
        elif engagement_score < self.thresholds['medium'] and self.get_engagement_trend() == "declining":
            return True, "declining_engagement"
        else:
            return False, "normal"
    
    def get_intervention_suggestions(self, intervention_type: str) -> List[str]:
        """Get suggestions for engagement intervention"""
        suggestions = {
            'low_engagement': [
                "Try a different learning approach",
                "Take a short break",
                "Use visual aids or examples",
                "Ask an engaging question",
                "Change the topic or difficulty level"
            ],
            'declining_engagement': [
                "Provide encouragement",
                "Simplify the current topic",
                "Use interactive elements",
                "Check if the user needs help",
                "Adjust the pace of interaction"
            ],
            'normal': [
                "Continue current approach",
                "Monitor engagement levels",
                "Maintain positive interaction"
            ]
        }
        
        return suggestions.get(intervention_type, ["Continue monitoring"])


def main():
    """Test engagement analyzer"""
    print("📊 Engagement Analyzer Test")
    print("=" * 40)
    
    analyzer = EngagementAnalyzer()
    
    # Simulate some signals
    print("Adding sample signals...")
    
    # Voice signals
    analyzer.add_voice_signal(800, 'happy', 0.8)
    analyzer.add_voice_signal(600, 'neutral', 0.7)
    analyzer.add_voice_signal(400, 'sad', 0.6)
    
    # Behavioral signals
    analyzer.add_behavioral_signal(2000, 0.8, 0.9)
    analyzer.add_behavioral_signal(3000, 0.6, 0.8)
    
    # Get summary
    summary = analyzer.get_engagement_summary()
    print(f"\n📈 Engagement Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test intervention
    should_intervene, intervention_type = analyzer.should_intervene(0.4)
    print(f"\n🚨 Intervention needed: {should_intervene} ({intervention_type})")
    
    if should_intervene:
        suggestions = analyzer.get_intervention_suggestions(intervention_type)
        print("💡 Suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")


if __name__ == "__main__":
    main()
