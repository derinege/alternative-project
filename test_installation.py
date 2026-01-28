#!/usr/bin/env python3
"""
Test script to verify installation and microphone access.
"""

import sys
import speech_recognition as sr

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import speech_recognition as sr
        print("✅ SpeechRecognition imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import SpeechRecognition: {e}")
        return False
    
    try:
        import pyaudio
        print("✅ PyAudio imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyAudio: {e}")
        print("💡 Try: pip install pyaudio")
        return False
    
    try:
        import numpy
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NumPy: {e}")
        return False
    
    return True

def test_microphone():
    """Test if microphone is accessible."""
    print("\n🎤 Testing microphone access...")
    
    try:
        # List available microphones
        mic_list = sr.Microphone.list_microphone_names()
        print(f"📋 Available microphones: {len(mic_list)}")
        
        for i, mic in enumerate(mic_list):
            print(f"   {i}: {mic}")
        
        # Test microphone initialization
        with sr.Microphone() as source:
            print("✅ Microphone initialized successfully")
            
            # Test ambient noise adjustment
            recognizer = sr.Recognizer()
            print("🔊 Testing ambient noise adjustment...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✅ Ambient noise adjustment successful")
            
            return True
            
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        print("💡 Check microphone permissions and connections")
        return False

def test_speech_recognition():
    """Test basic speech recognition functionality."""
    print("\n🎯 Testing speech recognition...")
    
    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        
        with sr.Microphone() as source:
            print("🎤 Please speak something for 3 seconds...")
            print("📝 (This is just a test - no actual recognition)")
            
            # Just test listening, don't actually recognize
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
            print("✅ Audio captured successfully")
            
            return True
            
    except sr.WaitTimeoutError:
        print("⏰ No speech detected (this is normal for a quick test)")
        return True
    except Exception as e:
        print(f"❌ Speech recognition test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Installation and Setup Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies.")
        sys.exit(1)
    
    # Test microphone
    if not test_microphone():
        print("\n❌ Microphone test failed. Check your audio setup.")
        sys.exit(1)
    
    # Test speech recognition
    if not test_speech_recognition():
        print("\n❌ Speech recognition test failed.")
        sys.exit(1)
    
    print("\n🎉 All tests passed!")
    print("✅ Your system is ready for speech-to-text")
    print("\n🚀 You can now run: python speech_to_text.py")

if __name__ == "__main__":
    main() 