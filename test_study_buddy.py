#!/usr/bin/env python3
"""
Test Script for Study Buddy Furhat
Verifies core functionality and system integration.

Author: Derin Ege Evren
Course: ELEC 491 Senior Design Project
"""

import sys
import os
import time
import numpy as np

def test_imports():
    """Test if all modules can be imported successfully"""
    print("🧪 Testing module imports...")
    
    try:
        from study_buddy import StudyBuddy
        from emotion_detector import EmotionDetector
        from engagement_analyzer import EngagementAnalyzer
        from dialogue_manager import DialogueManager
        from audio_processor import AudioProcessor
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_audio_processor():
    """Test audio processing functionality"""
    print("\n🎵 Testing audio processor...")
    
    try:
        from audio_processor import AudioProcessor
        
        # Create test audio signal
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = np.sin(2 * np.pi * 220 * t) * 0.5  # 220 Hz sine wave
        
        # Convert to bytes
        audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()
        
        # Test audio processor
        processor = AudioProcessor(sample_rate)
        features = processor.extract_features(audio_bytes)
        
        print(f"✅ Audio features extracted: {len(features)} features")
        print(f"   - RMS: {features.get('rms', 0):.3f}")
        print(f"   - Spectral centroid: {features.get('spectral_centroid', 0):.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Audio processor test failed: {e}")
        return False

def test_emotion_detector():
    """Test emotion detection functionality"""
    print("\n🎭 Testing emotion detector...")
    
    try:
        from emotion_detector import EmotionDetector
        
        detector = EmotionDetector()
        
        # Test voice emotion detection
        sample_audio_features = {
            'rms': 500,
            'dominant_freq': 150,
            'dynamic_range': 2.5,
            'zero_crossing_rate': 0.3
        }
        
        emotion = detector.detect_emotion_from_voice(sample_audio_features)
        print(f"✅ Emotion detected: {emotion}")
        
        # Test emotion classification
        is_positive = detector.is_positive_emotion(emotion)
        is_negative = detector.is_negative_emotion(emotion)
        
        print(f"   - Positive emotion: {is_positive}")
        print(f"   - Negative emotion: {is_negative}")
        
        return True
    except Exception as e:
        print(f"❌ Emotion detector test failed: {e}")
        return False

def test_engagement_analyzer():
    """Test engagement analysis functionality"""
    print("\n📊 Testing engagement analyzer...")
    
    try:
        from engagement_analyzer import EngagementAnalyzer
        
        analyzer = EngagementAnalyzer()
        
        # Add some test signals
        analyzer.add_voice_signal(800, 'happy', 0.8)
        analyzer.add_voice_signal(600, 'neutral', 0.7)
        analyzer.add_behavioral_signal(2000, 0.8, 0.9)
        
        # Test engagement calculation
        sample_audio_features = {
            'rms': 500,
            'dominant_freq': 150,
            'dynamic_range': 2.5,
            'zero_crossing_rate': 0.3
        }
        
        engagement = analyzer.calculate_engagement(sample_audio_features, 'happy', 'Hello, this is a test message')
        print(f"✅ Engagement calculated: {engagement:.3f}")
        
        # Test engagement summary
        summary = analyzer.get_engagement_summary()
        print(f"   - Overall engagement: {summary.get('overall_engagement', 0):.3f}")
        print(f"   - Engagement level: {summary.get('engagement_level', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Engagement analyzer test failed: {e}")
        return False

def test_dialogue_manager():
    """Test dialogue management functionality"""
    print("\n💬 Testing dialogue manager...")
    
    try:
        from dialogue_manager import DialogueManager
        
        manager = DialogueManager()
        
        # Test response generation (without LLM)
        user_input = "I'm having trouble understanding this concept"
        engagement_level = 0.3
        context = {
            'current_topic': 'mathematics',
            'session_duration': 600,
            'user_struggling': True,
            'use_llm': False  # Test without LLM first
        }
        
        response = manager.generate_response(user_input, engagement_level, None, context)
        
        print(f"✅ Response generated: {response['text'][:50]}...")
        print(f"   - Tone: {response['tone']}")
        print(f"   - State: {response['state']}")
        
        return True
    except Exception as e:
        print(f"❌ Dialogue manager test failed: {e}")
        return False

def test_study_buddy_core():
    """Test Study Buddy core functionality"""
    print("\n🤖 Testing Study Buddy core...")
    
    try:
        from study_buddy import StudyBuddy
        
        # Initialize Study Buddy
        buddy = StudyBuddy()
        
        # Test session start
        if buddy.start_session("test_user"):
            print("✅ Study session started successfully")
            
            # Test audio processing
            sample_audio = b'\x00\x01' * 8000  # Simple test audio data
            result = buddy.process_audio_input(sample_audio)
            
            if 'error' not in result:
                print(f"✅ Audio processing result: {len(result)} fields")
            else:
                print(f"⚠️ Audio processing error: {result['error']}")
            
            # Test response generation
            context = {
                'current_topic': 'general',
                'session_duration': 60,
                'interaction_count': 1,
                'use_llm': False
            }
            
            response = buddy.generate_response("Hello Study Buddy", context)
            print(f"✅ Response generated: {len(response)} fields")
            
            return True
        else:
            print("❌ Failed to start study session")
            return False
            
    except Exception as e:
        print(f"❌ Study Buddy core test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️ Testing configuration...")
    
    try:
        import json
        
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            print("✅ Configuration loaded successfully")
            print(f"   - Whisper model: {config.get('whisper_model', 'unknown')}")
            print(f"   - Target engagement: {config.get('target_engagement_score', 'unknown')}")
            print(f"   - Ollama endpoint: {config.get('ollama_endpoint', 'unknown')}")
            
            return True
        else:
            print("⚠️ Configuration file not found, using defaults")
            return True
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎓 Study Buddy Furhat - System Test Suite")
    print("=" * 50)
    print("ELEC 491 Senior Design Project")
    print("Derin Ege Evren | Prof. Engin Erzin")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Audio Processor", test_audio_processor),
        ("Emotion Detector", test_emotion_detector),
        ("Engagement Analyzer", test_engagement_analyzer),
        ("Dialogue Manager", test_dialogue_manager),
        ("Study Buddy Core", test_study_buddy_core),
        ("Configuration", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Study Buddy is ready to go!")
        print("\n🚀 To start Study Buddy:")
        print("   python study_buddy_app.py")
        print("\n🌐 Then visit: http://localhost:3000")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
    
    print("=" * 50)

if __name__ == "__main__":
    main()




