#!/usr/bin/env python3
"""
Study Buddy Furhat - Flask Web Application
Real-time engagement-aware conversational robot with multimodal interaction.

Author: Derin Ege Evren
Course: ELEC 491 Senior Design Project
Advisor: Prof. Engin Erzin
"""

import os
import time
import threading
import json
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import speech_recognition as sr
from faster_whisper import WhisperModel
import pyaudio
import wave
import io

# Study Buddy components
from study_buddy import StudyBuddy, InteractionState
from emotion_detector import EmotionDetector
from engagement_analyzer import EngagementAnalyzer
from dialogue_manager import DialogueManager
from audio_processor import AudioProcessor


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'study-buddy-secret-key-2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global Study Buddy instance
study_buddy = None
is_listening = False
recognizer = sr.Recognizer()

# Session management
current_session = {
    'user_id': None,
    'start_time': None,
    'interaction_count': 0,
    'total_engagement': 0.0
}


def initialize_study_buddy():
    """Initialize Study Buddy system"""
    global study_buddy
    try:
        study_buddy = StudyBuddy()
        print("🤖 Study Buddy system initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Study Buddy initialization failed: {e}")
        return False


@app.route('/')
def index():
    """Main Study Buddy interface"""
    return render_template('study_buddy.html')


@app.route('/dashboard')
def dashboard():
    """Engagement monitoring dashboard"""
    return render_template('dashboard.html')


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("🌐 New Study Buddy connection established")
    emit('connection_status', {'status': 'connected', 'message': 'Study Buddy is ready!'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("❌ Study Buddy connection lost")
    if current_session['user_id']:
        end_session()


@socketio.on('start_session')
def handle_start_session(data):
    """Start a new study session"""
    global current_session
    
    try:
        user_id = data.get('user_id', 'anonymous_user')
        
        if study_buddy and study_buddy.start_session(user_id):
            current_session['user_id'] = user_id
            current_session['start_time'] = time.time()
            current_session['interaction_count'] = 0
            current_session['total_engagement'] = 0.0
            
            print(f"🎓 Study session started for user: {user_id}")
            
            emit('session_started', {
                'user_id': user_id,
                'message': 'Study session started successfully!',
                'study_buddy_name': 'Study Buddy'
            })
            
            # Send initial greeting
            greeting = study_buddy.dialogue_manager._generate_greeting_response(0.5, {})
            emit('study_buddy_message', {
                'text': greeting,
                'tone': 'encouraging',
                'timestamp': time.time()
            })
            
        else:
            emit('session_error', {'error': 'Failed to start study session'})
            
    except Exception as e:
        print(f"❌ Session start error: {e}")
        emit('session_error', {'error': str(e)})


@socketio.on('end_session')
def end_session():
    """End current study session"""
    global current_session
    
    try:
        if current_session['user_id'] and study_buddy:
            # Get session summary
            summary = study_buddy.get_session_summary()
            
            print(f"🎓 Study session ended for user: {current_session['user_id']}")
            print(f"📊 Session summary: {summary}")
            
            emit('session_ended', {
                'user_id': current_session['user_id'],
                'summary': summary,
                'message': 'Study session completed!'
            })
            
            # Reset session
            current_session = {
                'user_id': None,
                'start_time': None,
                'interaction_count': 0,
                'total_engagement': 0.0
            }
            
        else:
            emit('session_error', {'error': 'No active session to end'})
            
    except Exception as e:
        print(f"❌ Session end error: {e}")
        emit('session_error', {'error': str(e)})


@socketio.on('start_listening')
def handle_start_listening():
    """Start listening for user speech"""
    global is_listening
    
    if not current_session['user_id']:
        emit('error', {'message': 'Please start a session first'})
        return
    
    is_listening = True
    print("🎤 Study Buddy started listening...")
    emit('listening_started', {'message': 'Listening for your input...'})


@socketio.on('stop_listening')
def handle_stop_listening():
    """Stop listening for user speech"""
    global is_listening
    
    is_listening = False
    print("⏹️ Study Buddy stopped listening")
    emit('listening_stopped', {'message': 'Stopped listening'})


@socketio.on('send_text_message')
def handle_text_message(data):
    """Handle text message from user"""
    try:
        if not current_session['user_id']:
            emit('error', {'message': 'Please start a session first'})
            return
        
        user_input = data.get('message', '').strip()
        if not user_input:
            return
        
        # Send user message to frontend first
        try:
            socketio.emit('user_message', {
                'text': user_input,
                'timestamp': time.time()
            })
        except Exception as e:
            print(f"⚠️ User message emit error: {e}")
        
        # Process text input
        context = {
            'current_topic': 'general',
            'session_duration': time.time() - current_session['start_time'],
            'interaction_count': current_session['interaction_count'],
            'use_llm': True
        }
        
        # Get current engagement level
        engagement_level = 0.6  # Default for text input
        
        # Generate response
        response = study_buddy.generate_response(user_input, context)
        
        # Update session
        current_session['interaction_count'] += 1
        current_session['total_engagement'] += response.get('engagement_score', 0.5)
        
        # Send response
        try:
            socketio.emit('study_buddy_message', {
                'text': response.get('text', 'I understand. Let me help you with that.'),
                'tone': response.get('tone', 'neutral'),
                'engagement_feedback': response.get('feedback', 'Good progress!'),
                'suggested_actions': response.get('actions', ['continue']),
                'timestamp': time.time()
            })
            
            # Send engagement update
            socketio.emit('engagement_update', {
                'engagement_score': response.get('engagement_score', 0.5),
                'level': study_buddy.engagement_analyzer.get_engagement_level(
                    response.get('engagement_score', 0.5)
                ),
                'timestamp': time.time()
            })
        except Exception as e:
            print(f"⚠️ Response emit error: {e}")
        
    except Exception as e:
        print(f"❌ Text message processing error: {e}")
        try:
            socketio.emit('error', {'message': 'Error processing your message'})
        except Exception as emit_error:
            print(f"⚠️ Error emit failed: {emit_error}")


def listen_and_process_audio():
    """Background thread for continuous audio processing"""
    global is_listening, current_session
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("🎤 Study Buddy audio system ready...")
        
        while True:
            if is_listening and current_session['user_id']:
                try:
                    print("🎤 Listening for speech...")
                    
                    # Listen for audio
                    audio = recognizer.listen(
                        source,
                        timeout=2.0,
                        phrase_time_limit=5.0
                    )
                    
                    audio_data = audio.get_raw_data()
                    
                    # Process audio with Study Buddy
                    result = study_buddy.process_audio_input(audio_data)
                    
                    if result.get('error'):
                        print(f"❌ Audio processing error: {result['error']}")
                        continue
                    
                    # Extract results
                    transcript = result.get('transcript', '')
                    emotion = result.get('emotion', 'neutral')
                    engagement_score = result.get('engagement_score', 0.5)
                    metrics = result.get('metrics', {})
                    
                    if transcript:
                        print(f"📝 User said: {transcript}")
                        print(f"🎭 Emotion: {emotion}")
                        print(f"📊 Engagement: {engagement_score:.3f}")
                        
                        # Send transcript to frontend
                        try:
                            with app.app_context():
                                socketio.emit('speech_transcript', {
                                    'text': transcript,
                                    'emotion': emotion,
                                    'confidence': metrics.get('confidence', 0.8),
                                    'timestamp': time.time()
                                })
                                socketio.emit('user_message', {
                                    'text': transcript,
                                    'timestamp': time.time()
                                })
                        except Exception as e:
                            print(f"⚠️ Transcript emit error: {e}")
                        
                        # Generate and send response
                        context = {
                            'current_topic': 'general',
                            'session_duration': time.time() - current_session['start_time'],
                            'interaction_count': current_session['interaction_count'],
                            'use_llm': True
                        }
                        
                        response = study_buddy.generate_response(transcript, context)
                        
                        # Update session
                        current_session['interaction_count'] += 1
                        current_session['total_engagement'] += engagement_score
                        
                        # Send Study Buddy response (with proper Flask context)
                        try:
                            with app.app_context():
                                socketio.emit('study_buddy_message', {
                                    'text': response.get('text', response.get('response_text', 'I understand. Let me help you with that.')),
                                    'tone': response.get('tone', response.get('emotional_tone', 'neutral')),
                                    'engagement_feedback': response.get('feedback', response.get('engagement_feedback', 'Good progress!')),
                                    'suggested_actions': response.get('actions', response.get('suggested_actions', ['continue'])),
                                    'timestamp': time.time()
                                })
                                
                                # Send engagement metrics
                                socketio.emit('engagement_metrics', {
                                    'engagement_score': engagement_score,
                                    'emotion': emotion,
                                    'voice_energy': metrics.get('voice_energy', 0),
                                    'attention_level': metrics.get('attention_level', 0),
                                    'response_time': metrics.get('response_time', 0),
                                    'interaction_quality': metrics.get('interaction_quality', 0),
                                    'timestamp': time.time()
                                })
                                
                                # Send engagement level update
                                socketio.emit('engagement_update', {
                                    'engagement_score': engagement_score,
                                    'level': study_buddy.engagement_analyzer.get_engagement_level(engagement_score),
                                    'trend': study_buddy.engagement_analyzer.get_engagement_trend(),
                                    'timestamp': time.time()
                                })
                        except Exception as e:
                            print(f"⚠️ Socket emit error: {e}")
                    
                except sr.WaitTimeoutError:
                    print("⏰ Audio timeout - no speech detected")
                except Exception as e:
                    print(f"❌ Audio processing error: {e}")
                    continue
            else:
                time.sleep(0.1)


@socketio.on('get_session_summary')
def handle_get_session_summary():
    """Get current session summary"""
    try:
        if current_session['user_id'] and study_buddy:
            summary = study_buddy.get_session_summary()
            emit('session_summary', summary)
        else:
            emit('session_error', {'error': 'No active session'})
    except Exception as e:
        print(f"❌ Session summary error: {e}")
        emit('session_error', {'error': str(e)})


@socketio.on('get_engagement_summary')
def handle_get_engagement_summary():
    """Get engagement analysis summary"""
    try:
        if study_buddy:
            summary = study_buddy.engagement_analyzer.get_engagement_summary()
            emit('engagement_summary', summary)
        else:
            emit('session_error', {'error': 'Study Buddy not initialized'})
    except Exception as e:
        print(f"❌ Engagement summary error: {e}")
        emit('session_error', {'error': str(e)})


@socketio.on('get_conversation_history')
def handle_get_conversation_history():
    """Get conversation history"""
    try:
        if study_buddy:
            history = study_buddy.dialogue_manager.get_conversation_summary()
            emit('conversation_history', history)
        else:
            emit('session_error', {'error': 'Study Buddy not initialized'})
    except Exception as e:
        print(f"❌ Conversation history error: {e}")
        emit('session_error', {'error': str(e)})


# Background thread for audio processing
def start_audio_thread():
    """Start the audio processing thread"""
    audio_thread = threading.Thread(target=listen_and_process_audio, daemon=True)
    audio_thread.start()
    print("🎵 Audio processing thread started")


if __name__ == '__main__':
    print("🎓 Study Buddy Furhat - Engagement-Aware Conversational Robot")
    print("=" * 70)
    print("📚 ELEC 491 Senior Design Project")
    print("👨‍🏫 Advisor: Prof. Engin Erzin")
    print("👨‍💻 Student: Derin Ege Evren")
    print("=" * 70)
    
    # Initialize Study Buddy
    if initialize_study_buddy():
        # Start audio processing thread
        start_audio_thread()
        
        print("🚀 Starting Study Buddy web server...")
        print("🌐 Access the interface at: http://localhost:3000")
        print("📊 Dashboard available at: http://localhost:3000/dashboard")
        print("💡 Make sure Ollama is running for enhanced dialogue!")
        
        # Start Flask-SocketIO server
        socketio.run(app, host='0.0.0.0', port=3000, debug=False)
    else:
        print("❌ Failed to initialize Study Buddy system")
        print("💡 Please check your configuration and dependencies")
