import os
import time
import threading
import requests
import json
import numpy as np
import re
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import speech_recognition as sr
from faster_whisper import WhisperModel
import pyaudio
import wave
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
is_listening = False
recognizer = sr.Recognizer()
transcript_text = ""
target_language = "en"
translation_service = "ollama"

# Whisper model yükle (iPhone 14 Pro için optimize edilmiş)
print("🤖 Whisper modeli yükleniyor...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Whisper modeli hazır! (iPhone 14 Pro için optimize edildi)")

# Ses analizi fonksiyonu
def analyze_audio(audio_data):
    """Ses sinyalini analiz et"""
    try:
        # Audio data'yı numpy array'e çevir
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Temel istatistikler
        rms = float(np.sqrt(np.mean(audio_array**2)))
        peak = float(np.max(np.abs(audio_array)))
        dynamic_range = float(peak / (rms + 1e-10))
        
        # Frekans analizi (basit)
        fft = np.fft.fft(audio_array)
        freqs = np.fft.fftfreq(len(audio_array), 1/16000)
        dominant_freq = float(abs(freqs[np.argmax(np.abs(fft))]))
        
        # SNR hesapla (basit)
        signal_power = float(np.mean(audio_array**2))
        noise_floor = float(np.percentile(audio_array**2, 10))
        # SNR hesaplaması güvenli hale getir
        if noise_floor < 1e-6:
            snr = float('nan')
        else:
            snr = float(10 * np.log10(signal_power / (noise_floor + 1e-10)))
        
        return {
            'rms': rms,
            'peak': peak,
            'dynamic_range': dynamic_range,
            'dominant_freq': dominant_freq,
            'snr': snr,
            'length_ms': float(len(audio_array) / 16)  # 16kHz sample rate
        }
    except Exception as e:
        print(f"❌ Ses analizi hatası: {e}")
        return None

def analyze_and_plot_audio(audio_data):
    """Ses sinyalini analiz et ve plot olarak kaydet"""
    try:
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        # Başlangıç spike'ını atla (ilk 1000 sample)
        if len(audio_array) > 1000:
            audio_array = audio_array[1000:]
        rms = float(np.sqrt(np.mean(audio_array**2)))
        peak = float(np.max(np.abs(audio_array)))
        dynamic_range = float(peak / (rms + 1e-10))
        fft = np.fft.fft(audio_array)
        freqs = np.fft.fftfreq(len(audio_array), 1/16000)
        dominant_freq = float(abs(freqs[np.argmax(np.abs(fft))]))
        signal_power = float(np.mean(audio_array**2))
        noise_floor = float(np.percentile(audio_array**2, 10))
        snr = float('nan') if noise_floor < 1e-6 else 10 * np.log10(signal_power / (noise_floor + 1e-10))
        # dB hesapla (referans: 32767)
        db = 20 * np.log10(rms / 32767 + 1e-10)
        # Plot ve kaydet
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        plt.figure(figsize=(10, 2))
        plt.plot(audio_array)
        plt.title('Audio Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(f'audio_waveform_{ts}.png')
        print(f'[AUDIO] Waveform plot saved: audio_waveform_{ts}.png')
        plt.close()
        # Spectrogram
        f, t, Sxx = spectrogram(audio_array, fs=16000)
        plt.figure(figsize=(10,4))
        plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-10), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogram')
        plt.colorbar(label='dB')
        plt.tight_layout()
        plt.savefig(f'audio_spectrogram_{ts}.png')
        print(f'[AUDIO] Spectrogram plot saved: audio_spectrogram_{ts}.png')
        plt.close()
        return {
            'rms': rms,
            'peak': peak,
            'snr': snr,
            'dominant_freq': dominant_freq,
            'db': db
        }
    except Exception as e:
        print(f'[AUDIO] Analysis error: {e}')
        return {}

# Çeviri cache'i - aynı kelimeleri tekrar çevirmemek için
translation_cache = {}

# Cümle algılama fonksiyonu
def detect_sentences(text):
    """Metni cümlelere ayırır ve noktalama işaretlerini düzeltir"""
    # Cümle sonu işaretlerini ekle
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Son cümleyi tamamla
    if text and not text[-1] in '.!?':
        if sentences:
            sentences[-1] = sentences[-1] + '.'
    
    return sentences

# Ollama ile hızlı çeviri
def translate_ollama(text, target_lang):
    """Ollama ile local çeviri - çok hızlı"""
    try:
        # Cache kontrolü
        cache_key = f"{text}_{target_lang}"
        if cache_key in translation_cache:
            return translation_cache[cache_key]
        
        # Dil kodlarını çevir
        lang_map = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic'
        }
        
        target_lang_name = lang_map.get(target_lang, 'English')
        
        # Ollama prompt'u
        prompt = f"""Translate this Turkish text to {target_lang_name}. Only return the translation, nothing else:

Turkish: {text}
{target_lang_name}:"""
        
        # Ollama ile çeviri
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2:1b",  # iPhone için daha hafif model
            "prompt": prompt,
            "stream": False
        }, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            translated = result.get('response', '').strip()
            
            # Cache'e kaydet
            translation_cache[cache_key] = translated
            
            return translated
        else:
            return f"Çeviri hatası: HTTP {response.status_code}"
        
    except Exception as e:
        print(f"Ollama çeviri hatası: {e}")
        return f"Çeviri hatası: {str(e)}"

# Google Translate API (fallback)
def translate_google(text, target_lang):
    """Google Translate ile çeviri (fallback)"""
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': 'tr',  # Kaynak dil: Türkçe
            'tl': target_lang,  # Hedef dil
            'dt': 't',
            'q': text
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json()
            translated_text = ''.join([part[0] for part in result[0] if part[0]])
            return translated_text
        else:
            return f"Çeviri hatası: {response.status_code}"
    except Exception as e:
        return f"Çeviri hatası: {str(e)}"

# Hızlı çeviri fonksiyonu
def translate_realtime(text, target_lang, service):
    """Gerçek zamanlı çeviri - çok hızlı"""
    try:
        if service == 'ollama':
            return translate_ollama(text, target_lang)
        elif service == 'google':
            return translate_google(text, target_lang)
        else:
            return "Desteklenmeyen çeviri servisi"
    except Exception as e:
        return f"Çeviri hatası: {str(e)}"

def translate_text(text, target_lang, service):
    """Metni çevir ve frontend'e gönder"""
    try:
        start_time = time.time()
        
        if service == "ollama":
            translated = translate_ollama(text, target_lang)
        else:
            translated = translate_google(text, target_lang)
        
        duration = (time.time() - start_time) * 1000  # ms cinsinden
        print(f"🌐 Çeviri: {translated} (Süre: {duration:.0f} ms)")
        
        # JSON serializable data gönder
        socketio.emit('translation_result', {
            'translated_text': translated,
            'duration': duration / 1000.0  # saniye cinsinden
        })
        
    except Exception as e:
        print(f"❌ Çeviri hatası: {e}")

def translate_with_ollama(text, target_lang):
    """Ollama ile çeviri"""
    try:
        # Ollama API endpoint
        url = "http://localhost:11434/api/generate"
        
        # Dil kodlarını Ollama formatına çevir
        lang_map = {
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese"
        }
        
        target_lang_name = lang_map.get(target_lang, "English")
        
        prompt = f"Translate the following Turkish text to {target_lang_name}. Only provide the translation, nothing else:\n\n{text}"
        
        data = {
            "model": "llama3.2:1b",  # iPhone için daha hafif model
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        translated = result.get('response', '').strip()
        
        return translated
        
    except Exception as e:
        print(f"❌ Ollama çeviri hatası: {e}")
        return f"[Çeviri hatası: {e}]"

def translate_with_google(text, target_lang):
    """Google Translate ile çeviri (fallback)"""
    try:
        # Basit Google Translate API simülasyonu
        # Gerçek uygulamada Google Translate API kullanılabilir
        return f"[Google: {text}]"
    except Exception as e:
        print(f"❌ Google çeviri hatası: {e}")
        return f"[Çeviri hatası: {e}]"

# Mikrofonları listele
@socketio.on('get_microphones')
def handle_get_microphones():
    try:
        mic_list = sr.Microphone.list_microphone_names()
        # Aktif mikrofon index'i (varsayılan 0)
        default_index = 0
        socketio.emit('microphone_list', {
            'microphones': [
                {'index': i, 'name': name} for i, name in enumerate(mic_list)
            ],
            'default_index': default_index
        })
    except Exception as e:
        socketio.emit('microphone_list', {'microphones': [], 'default_index': 0, 'error': str(e)})

def listen_and_transcribe():
    global is_listening, transcript_text
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        print("🎤 Sistem hazır. Dinleme başlatılmayı bekliyor...")
        
        while True:
            if is_listening:
                try:
                    print("🎤 Dinleniyor...")
                    audio_start_time = time.time()
                    
                    # iPhone 14 Pro için optimize edilmiş audio kayıt parametreleri
                    audio = recognizer.listen(
                        source, 
                        timeout=3.0,  # iPhone için daha kısa timeout
                        phrase_time_limit=3.0  # iPhone için daha kısa phrase limit
                    )
                    audio_data = audio.get_raw_data()
                    
                    # Sinyal analizi ve dB hesapla
                    features = analyze_and_plot_audio(audio_data)
                    
                    # dB seviyesini frontend'e gönder
                    socketio.emit('audio_features', {'db': features.get('db', None)})
                    
                    # WAV buffer oluştur
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(audio_data)
                    wav_buffer.seek(0)
                    
                    # Whisper ile transcribe
                    print("🔄 Whisper ile işleniyor...")
                    stt_start = time.time()
                    # iPhone 14 Pro için optimize edilmiş Whisper parametreleri
                    segments, info = whisper_model.transcribe(
                        wav_buffer,
                        beam_size=1,  # En hızlı
                        temperature=0.0,  # Deterministik
                        log_prob_threshold=-1.0,
                        no_speech_threshold=0.4,  # iPhone için biraz daha hassas
                        language="tr",  # Türkçe zorlaması
                        condition_on_previous_text=False,  # Hız için
                        initial_prompt="Aşağıdaki Türkçe konuşmayı doğru ve eksiksiz yazıya dök. Kısaltma, atlama veya değiştirme yapma. Sadece konuşulanı yaz.",
                        word_timestamps=False,  # Hız için
                        max_initial_timestamp=0.5,  # iPhone için daha kısa
                        max_new_tokens=32  # iPhone için daha kısa çıktı
                    )
                    stt_end = time.time()
                    
                    # Segmentleri birleştir
                    text = ""
                    for segment in segments:
                        text += segment.text + " "
                    
                    text = text.strip()
                    
                    if text:
                        stt_duration = (stt_end - stt_start) * 1000  # ms cinsinden
                        print(f"📝 Whisper: {text} (Süre: {stt_duration:.0f} ms)")
                        
                        # Algılanan dili belirle
                        detected_lang = "tr"  # Türkçe zorlaması
                        
                        # Frontend'e gönder - JSON serializable data
                        socketio.emit('new_text', {
                            'text': text,
                            'stt_duration': stt_duration / 1000.0,  # saniye cinsinden
                            'lang': detected_lang
                        })
                        
                        # Dil algılama bilgisini gönder
                        if detected_lang:
                            socketio.emit('detected_language', {
                                'lang': detected_lang
                            })
                        
                        # Çeviri yap
                        if target_language and translation_service:
                            translate_text(text, target_language, translation_service)
                    
                except sr.WaitTimeoutError:
                    print("⏰ Dinleme zaman aşımı")
                except Exception as e:
                    print(f"Dinleme/Transkripsiyon hatası: {e}")
                    continue
            else:
                time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("🌐 Yeni bağlantı başarılı")

@socketio.on('disconnect')
def handle_disconnect():
    print("❌ Bağlantı kesildi")

@socketio.on('start_listening')
def handle_start_listening(data):
    global is_listening, target_language, translation_service
    is_listening = True
    target_language = data.get('target_lang', 'en')
    translation_service = data.get('service', 'ollama')
    print(f"🎤 Dinleme başlatıldı - Hedef dil: {target_language}, Servis: {translation_service}")
    socketio.emit('listening_started')

@socketio.on('stop_listening')
def handle_stop_listening():
    global is_listening, transcript_text
    is_listening = False
    print("⏹️ Dinleme durduruldu")
    print(f"📄 Tam transkript: {transcript_text}")
    socketio.emit('listening_stopped')

# Background thread başlat
threading.Thread(target=listen_and_transcribe, daemon=True).start()

if __name__ == '__main__':
    # Arka planda konuşma tanıma thread'i başlat
    # t = threading.Thread(target=recognize_speech_background, daemon=True)
    # t.start()
    print("🚀 Sunucu başlatılıyor... http://localhost:3000")
    print("⚡ Ollama ile hızlı çeviri aktif!")
    # Flask-SocketIO sunucusunu başlat
    socketio.run(app, host='0.0.0.0', port=3000, debug=False) 