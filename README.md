# Real-Time Speech Translation System (ELEC_491)

Bu proje, gerçek zamanlı konuşma tanıma ve çeviri yapan, taşınabilir ve mobil uyumlu bir sistemdir. Amaç, yaka mikrofonu veya telefon mikrofonundan alınan sesi anında yazıya dökmek ve seçilen dile çevirmektir. Tüm süreç local olarak çalışır, internet gerektirmez ve modern bir web arayüzü sunar.

## Özellikler

- 🎤 **Gerçek zamanlı konuşma tanıma** (Whisper - local, hızlı, çok dilli)
- 🌐 **Anında çeviri** (Ollama LLM - local, hızlı, gizli)
- 📱 **Mobil uyumlu** (iPhone 14 Pro ve üstü, MacBook, taşınabilir sistemler)
- 🖥️ **Web arayüzü** (canlı dB seviyesi, transkript, çeviri, dil algılama)
- 🔊 **dB seviyesi ve sinyal analizi** (canlı görsel bar)
- 🛠️ **Kolay konfigürasyon** (hedef dil, çeviri servisi seçimi)
- 🔒 **Gizlilik** (tüm veriler localde işlenir)

## Gereksinimler

- Python 3.8+
- macOS veya Linux (test: MacBook, iPhone)
- [Ollama](https://ollama.com/) (local LLM için)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (local STT için)
- Mikrofon erişimi

## Kurulum

1. **Projeyi klonlayın:**
   ```bash
   git clone <repo-url>
   cd ELEC_491
   ```
2. **Sanal ortam oluşturun ve bağımlılıkları yükleyin:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Ollama modelini indirin:**
   ```bash
   ollama pull llama3.2:1b
   ollama run llama3.2:1b
   ```
4. **Sunucuyu başlatın:**
   ```bash
   python app.py
   ```
5. **Web arayüzüne girin:**
   - [http://localhost:3000](http://localhost:3000)

## Kullanım

- "Dinlemeyi Başlat" butonuna tıklayın.
- Konuşun, sistem otomatik olarak yazıya döker ve çevirir.
- dB barı ile ses seviyenizi canlı izleyin.
- Hedef dili ve çeviri servisini seçebilirsiniz.

## Sistem Mimarisi

```
Mikrofon → Whisper (STT) → Transkript → Ollama (LLM) → Çeviri → Web Arayüzü
```

## Teknik Detaylar

- **Whisper (faster-whisper):**
  - Model: `base` (mobil için optimize, hızlı ve doğru)
  - initial_prompt ile doğruluk artırıldı
  - Türkçe zorlaması ve otomatik dil algılama
- **Ollama (Llama 3.2:1b):**
  - Local LLM ile hızlı çeviri
  - Hedef dil seçilebilir
- **Web Arayüzü:**
  - Canlı dB barı, transkript, çeviri, dil algılama
  - Modern ve mobil uyumlu tasarım

## Sık Karşılaşılan Sorunlar

- **HTTP 404 Çeviri Hatası:** Ollama modelinin tam yüklendiğinden ve çalıştığından emin olun.
- **Yanlış Transkript:** Mikrofonu yaklaştırın, Whisper modelini `base` veya daha üstü yapın.
- **dB çok düşük:** Daha yüksek sesle konuşun veya mikrofonu değiştirin.

## Geliştirme ve Katkı

- Kodlarınızı ve iyileştirmelerinizi paylaşabilirsiniz.
- Donanım entegrasyonu (yaka mikrofonu, ESP32, vb.) için önerilere açıktır.

## Lisans

Bu proje ELEC_491 dersi kapsamında geliştirilmiştir. 