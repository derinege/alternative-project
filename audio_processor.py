#!/usr/bin/env python3
"""
Audio Processing Module for Study Buddy
Enhanced audio feature extraction and analysis for emotion and engagement detection.

Author: Derin Ege Evren
Course: ELEC 491 Senior Design Project
"""

import numpy as np
import librosa
import scipy.signal
from typing import Dict, List, Tuple, Optional
import wave
import io


class AudioProcessor:
    """
    Advanced audio processing for Study Buddy's multimodal analysis
    """
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize audio processor"""
        self.sample_rate = sample_rate
        
        # Audio feature extraction parameters
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        
        print(f"🎵 Audio processor initialized (SR: {sample_rate} Hz)")
    
    def extract_features(self, audio_data: bytes) -> Dict:
        """Extract comprehensive audio features from raw audio data"""
        try:
            # Convert bytes to numpy array
            audio_array = self._bytes_to_audio_array(audio_data)
            
            if len(audio_array) == 0:
                return self._empty_features()
            
            # Basic audio features
            basic_features = self._extract_basic_features(audio_array)
            
            # Spectral features
            spectral_features = self._extract_spectral_features(audio_array)
            
            # Prosodic features
            prosodic_features = self._extract_prosodic_features(audio_array)
            
            # Voice activity detection
            vad_features = self._extract_vad_features(audio_array)
            
            # Combine all features
            features = {
                **basic_features,
                **spectral_features,
                **prosodic_features,
                **vad_features
            }
            
            return features
            
        except Exception as e:
            print(f"❌ Audio feature extraction error: {e}")
            return self._empty_features()
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert raw audio bytes to numpy array"""
        try:
            # Assume 16-bit PCM audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float and normalize
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Remove initial spike if present
            if len(audio_array) > 1000:
                audio_array = audio_array[1000:]
            
            return audio_array
            
        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            return np.array([])
    
    def _extract_basic_features(self, audio: np.ndarray) -> Dict:
        """Extract basic audio features"""
        if len(audio) == 0:
            return self._empty_basic_features()
        
        # RMS energy
        rms = float(np.sqrt(np.mean(audio**2)))
        
        # Peak amplitude
        peak = float(np.max(np.abs(audio)))
        
        # Dynamic range
        dynamic_range = float(peak / (rms + 1e-10))
        
        # Zero crossing rate
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)))
        
        # Signal-to-noise ratio (simplified)
        signal_power = np.mean(audio**2)
        noise_floor = np.percentile(audio**2, 10)
        snr = float(10 * np.log10(signal_power / (noise_floor + 1e-10))) if noise_floor > 1e-10 else 0.0
        
        return {
            'rms': rms,
            'peak': peak,
            'dynamic_range': dynamic_range,
            'zero_crossing_rate': zcr,
            'snr': snr,
            'length_ms': len(audio) / self.sample_rate * 1000
        }
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict:
        """Extract spectral features"""
        if len(audio) == 0:
            return self._empty_spectral_features()
        
        try:
            # Short-time Fourier transform
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            # Spectral centroid
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate)))
            
            # Spectral rolloff
            spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate)))
            
            # Spectral bandwidth
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate)))
            
            # Dominant frequency
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
            dominant_freq = float(freqs[np.argmax(np.mean(magnitude, axis=1))])
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=self.sample_rate)
            spectral_contrast_mean = float(np.mean(spectral_contrast))
            
            return {
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'dominant_freq': dominant_freq,
                'spectral_contrast': spectral_contrast_mean
            }
            
        except Exception as e:
            print(f"❌ Spectral feature extraction error: {e}")
            return self._empty_spectral_features()
    
    def _extract_prosodic_features(self, audio: np.ndarray) -> Dict:
        """Extract prosodic features (pitch, rhythm, etc.)"""
        if len(audio) == 0:
            return self._empty_prosodic_features()
        
        try:
            # Fundamental frequency (pitch)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
            
            # Extract pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                mean_pitch = float(np.mean(pitch_values))
                pitch_std = float(np.std(pitch_values))
                pitch_range = float(np.max(pitch_values) - np.min(pitch_values))
            else:
                mean_pitch = 0.0
                pitch_std = 0.0
                pitch_range = 0.0
            
            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            tempo = float(tempo) if not np.isnan(tempo) else 0.0
            
            # Rhythm regularity
            rhythm_regularity = self._calculate_rhythm_regularity(audio)
            
            return {
                'mean_pitch': mean_pitch,
                'pitch_std': pitch_std,
                'pitch_range': pitch_range,
                'tempo': tempo,
                'rhythm_regularity': rhythm_regularity
            }
            
        except Exception as e:
            print(f"❌ Prosodic feature extraction error: {e}")
            return self._empty_prosodic_features()
    
    def _extract_vad_features(self, audio: np.ndarray) -> Dict:
        """Extract voice activity detection features"""
        if len(audio) == 0:
            return {'voice_activity': False, 'speech_ratio': 0.0}
        
        try:
            # Simple VAD based on energy threshold
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length//2)
            
            # Calculate energy for each frame
            frame_energies = np.sum(frames**2, axis=0)
            
            # Dynamic threshold (percentile-based)
            energy_threshold = np.percentile(frame_energies, 30)
            
            # Voice activity detection
            voice_frames = frame_energies > energy_threshold
            speech_ratio = float(np.sum(voice_frames) / len(voice_frames))
            
            # Voice activity decision
            voice_activity = speech_ratio > 0.3
            
            # Speech rate (approximate)
            speech_rate = self._estimate_speech_rate(voice_frames)
            
            return {
                'voice_activity': voice_activity,
                'speech_ratio': speech_ratio,
                'speech_rate': speech_rate,
                'energy_threshold': float(energy_threshold)
            }
            
        except Exception as e:
            print(f"❌ VAD feature extraction error: {e}")
            return {'voice_activity': False, 'speech_ratio': 0.0, 'speech_rate': 0.0}
    
    def _calculate_rhythm_regularity(self, audio: np.ndarray) -> float:
        """Calculate rhythm regularity (simplified)"""
        try:
            # Onset detection
            onsets = librosa.onset.onset_detect(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
            
            if len(onsets) < 3:
                return 0.0
            
            # Calculate inter-onset intervals
            iois = np.diff(onsets) / self.sample_rate * self.hop_length
            
            # Calculate regularity as inverse of coefficient of variation
            if np.mean(iois) > 0:
                cv = np.std(iois) / np.mean(iois)
                regularity = 1.0 / (1.0 + cv)
            else:
                regularity = 0.0
            
            return float(regularity)
            
        except Exception as e:
            print(f"❌ Rhythm regularity calculation error: {e}")
            return 0.0
    
    def _estimate_speech_rate(self, voice_frames: np.ndarray) -> float:
        """Estimate speech rate from voice activity"""
        try:
            # Count speech segments
            speech_segments = []
            in_speech = False
            segment_start = 0
            
            for i, is_speech in enumerate(voice_frames):
                if is_speech and not in_speech:
                    segment_start = i
                    in_speech = True
                elif not is_speech and in_speech:
                    speech_segments.append(i - segment_start)
                    in_speech = False
            
            if in_speech:
                speech_segments.append(len(voice_frames) - segment_start)
            
            # Estimate speech rate (words per minute approximation)
            if speech_segments:
                avg_segment_length = np.mean(speech_segments)
                # Rough conversion: longer segments = slower speech
                speech_rate = 200.0 / (avg_segment_length + 1)  # WPM approximation
            else:
                speech_rate = 0.0
            
            return float(speech_rate)
            
        except Exception as e:
            print(f"❌ Speech rate estimation error: {e}")
            return 0.0
    
    def extract_emotion_features(self, audio_features: Dict) -> Dict:
        """Extract features specifically relevant for emotion detection"""
        try:
            emotion_features = {}
            
            # Energy-based features
            emotion_features['energy_level'] = audio_features.get('rms', 0)
            emotion_features['energy_variance'] = audio_features.get('dynamic_range', 1)
            
            # Pitch-based features
            emotion_features['pitch_level'] = audio_features.get('mean_pitch', 0)
            emotion_features['pitch_variance'] = audio_features.get('pitch_std', 0)
            emotion_features['pitch_range'] = audio_features.get('pitch_range', 0)
            
            # Spectral features
            emotion_features['brightness'] = audio_features.get('spectral_centroid', 0)
            emotion_features['spectral_contrast'] = audio_features.get('spectral_contrast', 0)
            
            # Rhythm features
            emotion_features['tempo'] = audio_features.get('tempo', 0)
            emotion_features['rhythm_regularity'] = audio_features.get('rhythm_regularity', 0)
            
            # Voice quality features
            emotion_features['voice_activity'] = audio_features.get('voice_activity', False)
            emotion_features['speech_rate'] = audio_features.get('speech_rate', 0)
            
            return emotion_features
            
        except Exception as e:
            print(f"❌ Emotion feature extraction error: {e}")
            return {}
    
    def extract_engagement_features(self, audio_features: Dict) -> Dict:
        """Extract features specifically relevant for engagement detection"""
        try:
            engagement_features = {}
            
            # Vocal energy and clarity
            engagement_features['vocal_energy'] = audio_features.get('rms', 0)
            engagement_features['speech_clarity'] = audio_features.get('snr', 0)
            
            # Interaction patterns
            engagement_features['speech_activity'] = audio_features.get('speech_ratio', 0)
            engagement_features['speech_rate'] = audio_features.get('speech_rate', 0)
            
            # Vocal dynamics
            engagement_features['pitch_variation'] = audio_features.get('pitch_std', 0)
            engagement_features['rhythm_consistency'] = audio_features.get('rhythm_regularity', 0)
            
            # Spectral characteristics
            engagement_features['spectral_brightness'] = audio_features.get('spectral_centroid', 0)
            engagement_features['spectral_richness'] = audio_features.get('spectral_contrast', 0)
            
            return engagement_features
            
        except Exception as e:
            print(f"❌ Engagement feature extraction error: {e}")
            return {}
    
    def _empty_features(self) -> Dict:
        """Return empty feature dictionary"""
        return {
            'rms': 0.0,
            'peak': 0.0,
            'dynamic_range': 1.0,
            'zero_crossing_rate': 0.0,
            'snr': 0.0,
            'length_ms': 0.0,
            'spectral_centroid': 0.0,
            'spectral_rolloff': 0.0,
            'spectral_bandwidth': 0.0,
            'dominant_freq': 0.0,
            'spectral_contrast': 0.0,
            'mean_pitch': 0.0,
            'pitch_std': 0.0,
            'pitch_range': 0.0,
            'tempo': 0.0,
            'rhythm_regularity': 0.0,
            'voice_activity': False,
            'speech_ratio': 0.0,
            'speech_rate': 0.0
        }
    
    def _empty_basic_features(self) -> Dict:
        """Return empty basic features"""
        return {
            'rms': 0.0,
            'peak': 0.0,
            'dynamic_range': 1.0,
            'zero_crossing_rate': 0.0,
            'snr': 0.0,
            'length_ms': 0.0
        }
    
    def _empty_spectral_features(self) -> Dict:
        """Return empty spectral features"""
        return {
            'spectral_centroid': 0.0,
            'spectral_rolloff': 0.0,
            'spectral_bandwidth': 0.0,
            'dominant_freq': 0.0,
            'spectral_contrast': 0.0
        }
    
    def _empty_prosodic_features(self) -> Dict:
        """Return empty prosodic features"""
        return {
            'mean_pitch': 0.0,
            'pitch_std': 0.0,
            'pitch_range': 0.0,
            'tempo': 0.0,
            'rhythm_regularity': 0.0
        }


def main():
    """Test audio processor"""
    print("🎵 Audio Processor Test")
    print("=" * 40)
    
    processor = AudioProcessor()
    
    # Generate test audio signal
    duration = 2.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple test signal (sine wave with some noise)
    frequency = 220  # A3 note
    test_audio = np.sin(2 * np.pi * frequency * t) * 0.5
    test_audio += np.random.normal(0, 0.1, len(test_audio))  # Add noise
    
    # Convert to bytes (simulating raw audio input)
    audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()
    
    # Extract features
    print("Extracting audio features...")
    features = processor.extract_features(audio_bytes)
    
    print(f"\n📊 Basic Features:")
    basic_features = ['rms', 'peak', 'dynamic_range', 'zero_crossing_rate', 'snr']
    for feature in basic_features:
        print(f"  {feature}: {features.get(feature, 0):.3f}")
    
    print(f"\n🌊 Spectral Features:")
    spectral_features = ['spectral_centroid', 'spectral_rolloff', 'dominant_freq', 'spectral_contrast']
    for feature in spectral_features:
        print(f"  {feature}: {features.get(feature, 0):.3f}")
    
    print(f"\n🎼 Prosodic Features:")
    prosodic_features = ['mean_pitch', 'pitch_std', 'tempo', 'rhythm_regularity']
    for feature in prosodic_features:
        print(f"  {feature}: {features.get(feature, 0):.3f}")
    
    print(f"\n🎤 VAD Features:")
    vad_features = ['voice_activity', 'speech_ratio', 'speech_rate']
    for feature in vad_features:
        print(f"  {feature}: {features.get(feature, 0)}")
    
    # Test emotion and engagement features
    emotion_features = processor.extract_emotion_features(features)
    engagement_features = processor.extract_engagement_features(features)
    
    print(f"\n🎭 Emotion Features: {len(emotion_features)} extracted")
    print(f"🎯 Engagement Features: {len(engagement_features)} extracted")


if __name__ == "__main__":
    main()




