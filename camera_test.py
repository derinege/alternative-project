#!/usr/bin/env python3
"""
Kamera Test Script - Study Buddy için kamera erişimini test eder
"""

import cv2
import sys

def test_camera():
    print("🎥 Kamera testi başlatılıyor...")
    
    # Kamera bağlantısını test et
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        print("💡 macOS'ta kamera izni vermen gerekiyor:")
        print("   1. System Preferences > Security & Privacy > Camera")
        print("   2. Terminal veya Python'a kamera izni ver")
        return False
    
    print("✅ Kamera bağlantısı başarılı!")
    
    # Kameradan bir frame al
    ret, frame = cap.read()
    if ret:
        print("✅ Kameradan görüntü alındı!")
        print(f"📐 Görüntü boyutu: {frame.shape[1]}x{frame.shape[0]}")
        
        # Yüz algılama test et
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            print(f"👤 Algılanan yüz sayısı: {len(faces)}")
            
            if len(faces) > 0:
                print("🎭 Yüz algılama başarılı! Study Buddy yüz ifadelerini analiz edebilir.")
                return True
            else:
                print("⚠️ Şu anda yüz algılanmadı. Kameranın önünde durun.")
                return True  # Kamera çalışıyor, sadece yüz yok
        except Exception as e:
            print(f"⚠️ Yüz algılama hatası: {e}")
            return True  # Kamera çalışıyor
    else:
        print("❌ Kameradan görüntü alınamadı!")
        return False
    
    cap.release()

if __name__ == "__main__":
    success = test_camera()
    if success:
        print("\n🎉 Kamera testi başarılı! Study Buddy kamera kullanabilir.")
    else:
        print("\n❌ Kamera testi başarısız. İzinleri kontrol edin.")




