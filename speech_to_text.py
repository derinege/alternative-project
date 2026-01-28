#!/usr/bin/env python3
"""
Real-time Speech-to-Text for Earbud Translation System
Supports Turkish language with noise adjustment and timeout handling.
"""

import speech_recognition as sr
import time
import sys
from typing import Optional


class RealTimeSpeechToText:
    def __init__(self, language: str = "tr-TR", timeout: int = 5, phrase_time_limit: int = 10):
        """
        Initialize the speech-to-text system.
        
        Args:
            language: Language code (default: Turkish)
            timeout: Timeout for listening in seconds
            phrase_time_limit: Maximum phrase length in seconds
        """
        self.recognizer = sr.Recognizer()
        self.language = language
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        
        # Adjust for ambient noise and microphone sensitivity
        self.recognizer.energy_threshold = 300  # Lower threshold for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Shorter pause for real-time feel
        
        # Noise adjustment settings
        self.recognizer.operation_timeout = None
        self.recognizer.non_speaking_duration = 0.5
        
        print(f"🎤 Speech-to-Text System Initialized")
        print(f"📝 Language: {language}")
        print(f"⏱️  Timeout: {timeout}s, Phrase limit: {phrase_time_limit}s")
        print(f"🔊 Energy threshold: {self.recognizer.energy_threshold}")
        print("=" * 50)
    
    def adjust_for_ambient_noise(self, duration: int = 1) -> None:
        """
        Adjust the recognizer for ambient noise.
        
        Args:
            duration: Duration to listen for ambient noise
        """
        print(f"🔊 Adjusting for ambient noise ({duration}s)...")
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                print(f"✅ Ambient noise adjustment complete")
                print(f"📊 New energy threshold: {self.recognizer.energy_threshold}")
        except Exception as e:
            print(f"⚠️  Warning: Could not adjust for ambient noise: {e}")
    
    def listen_and_transcribe(self) -> Optional[str]:
        """
        Listen to microphone and transcribe speech.
        
        Returns:
            Transcribed text or None if no speech detected
        """
        try:
            with sr.Microphone() as source:
                print("🎤 Listening... (speak now)")
                
                # Listen for audio input
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit
                )
                
                print("🔄 Processing speech...")
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(
                    audio,
                    language=self.language
                )
                
                return text.strip()
                
        except sr.WaitTimeoutError:
            print("⏰ Timeout: No speech detected")
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"🌐 Network error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def run_continuous(self) -> None:
        """
        Run continuous speech-to-text listening.
        """
        print("🚀 Starting continuous speech recognition...")
        print("💡 Press Ctrl+C to stop")
        print("=" * 50)
        
        try:
            while True:
                text = self.listen_and_transcribe()
                
                if text:
                    print(f"📝 Transcribed: {text}")
                    print("-" * 30)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping speech recognition...")
            print("👋 Goodbye!")


def main():
    """Main function to run the speech-to-text system."""
    print("🎧 Real-Time Speech-to-Text for Earbud Translation System")
    print("=" * 60)
    
    # Initialize the speech-to-text system
    stt = RealTimeSpeechToText(
        language="tr-TR",  # Turkish language
        timeout=5,         # 5 second timeout
        phrase_time_limit=10  # 10 second phrase limit
    )
    
    # Adjust for ambient noise
    stt.adjust_for_ambient_noise(duration=2)
    
    # Start continuous listening
    stt.run_continuous()


if __name__ == "__main__":
    main() 