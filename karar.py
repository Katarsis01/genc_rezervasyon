import re
import joblib
import os
import pandas as pd
import random
from difflib import SequenceMatcher

MODEL_PATH = "kyk_rezervasyon_model.pkl"
VECTORIZER_PATH = "kyk_rezervasyon_vectorizer.pkl"
KELIME_PUAN_DOSYA = "kelime_puanlari.txt"
TRAIN_DOSYA = "basvuru_train.csv"

stopwords = set([
    "ve", "bir", "bu", "ile", "de", "da", "için", "mi", "ne", "ya", "ama",
    "çok", "daha", "en", "gibi", "ise", "veya", "şu", "o", "ki",
    "ben", "sen", "biz", "siz", "onlar", "olarak", "yani", "ancak", "hem",
    "çünkü", "fakat", "her", "hiç", "nasıl", "neden", "niçin", "hangi", "kim"
])

def temizle_ve_parcala(cumle):
    cumle = cumle.lower()
    kelimeler = re.findall(r'\b\w+\b', cumle)
    anlamli = [k for k in kelimeler if k not in stopwords and len(k) > 2]
    return anlamli

def benzer_mi(k1, k2, thresh=0.7):
    if k1[0] != k2[0]:
        return False
    oran = SequenceMatcher(None, k1, k2).ratio()
    return oran >= thresh

def kelime_puan_yukle(dosya):
    puanlar = dict()
    with open(dosya, "r", encoding="utf-8") as f:
        for satir in f:
            parca = satir.strip().split(",")
            if len(parca) == 2:
                kelime, puan = parca
                puanlar[kelime] = float(puan)
    return puanlar

def kelime_puan_hesapla(metin, puan_tablosu):
    kelimeler = temizle_ve_parcala(metin)
    toplam_puan = 0.0
    eslesen = 0

    for kelime in kelimeler:
        for listedeki in puan_tablosu:
            if benzer_mi(kelime, listedeki):
                toplam_puan += puan_tablosu[listedeki]
                eslesen += 1
                break

    if eslesen == 0:
        return 0.0

    ortalama_puan = toplam_puan / eslesen
    max_puan = max(puan_tablosu.values())
    min_puan = min(puan_tablosu.values())
    if max_puan == min_puan:
        normalized = 5.0  # Ortalamadan kabul edelim
    else:
        normalized = (ortalama_puan - min_puan) / (max_puan - min_puan) * 10

    final_score =(3 + normalized ) * 0.5
    return final_score


def yukle_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model dosyaları eksik!")
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def yorumla_10_uzerinden(skor):
    if skor < 2:
        return "Çok Düşük Öncelik"
    elif skor < 4:
        return "Düşük Öncelik"
    elif skor < 6:
        return "Orta Öncelik"
    elif skor < 8:
        return "Yüksek Öncelik"
    else:
        return "Çok Yüksek Öncelik"

def hybrid_tahmin_ham(metin, model, vectorizer, kelime_puan_tablosu):
    vec = vectorizer.transform([metin])
    rf_tahmin = model.predict(vec)[0]
    rf_skor = max(0.0, min(rf_tahmin, 10.0)) * 0.5
    kelime_skor = kelime_puan_hesapla(metin, kelime_puan_tablosu)
    toplam = round(rf_skor + kelime_skor, 3)
    return toplam, yorumla_10_uzerinden(toplam)

def hybrid_tahmin_yap(metin):
    model, vectorizer = yukle_model()
    if model is None:
        return

    kelime_puan_tablosu = kelime_puan_yukle(KELIME_PUAN_DOSYA)
    
    # Skorların alt detayları
    vec = vectorizer.transform([metin])
    rf_raw = model.predict(vec)[0]
    rf_skor = max(0.0, min(rf_raw, 10.0)) * 0.4
    kelime_skor = kelime_puan_hesapla(metin, kelime_puan_tablosu)
    toplam = round(rf_skor + kelime_skor, 3)
    yorum = yorumla_10_uzerinden(toplam)

    print(f"\nToplam Skor: {toplam:.2f} / 10.00 → {yorum}")
    print(f"  ⮑ Random Forest katkısı: {rf_skor:.2f} (ham tahmin: {rf_raw:.2f})")
    print(f"  ⮑ Kelime Puanlama katkısı: {kelime_skor:.2f}")


def toplu_degerlendirme():
    model, vectorizer = yukle_model()
    kelime_puan_tablosu = kelime_puan_yukle(KELIME_PUAN_DOSYA)
    if model is None or not os.path.exists(TRAIN_DOSYA):
        print("Model veya eğitim dosyası eksik.")
        return

    df = pd.read_csv(TRAIN_DOSYA).dropna(subset=["başvuru_metni", "öncelik_skoru"])
    örnekler = df.sample(n=10, random_state=42)

    print("\n--- Toplu Değerlendirme (10 örnek) ---")
    for i, row in örnekler.iterrows():
        cumle = row["başvuru_metni"]
        gercek = row["öncelik_skoru"]
        
        vec = vectorizer.transform([cumle])
        rf_raw = model.predict(vec)[0]
        rf_skor = max(0.0, min(rf_raw, 10.0)) * 0.5
        kelime_skor = kelime_puan_hesapla(cumle, kelime_puan_tablosu)
        toplam = round(rf_skor + kelime_skor, 3)
        yorum = yorumla_10_uzerinden(toplam)

        print("\n--- Örnek ---")
        print(f"Metin: {cumle}")
        print(f"Gerçek Skor: {gercek:.2f}")
        print(f"Tahmini Skor: {toplam:.2f} → {yorum}")
        print(f"  ⮑ Random Forest katkısı: {rf_skor:.2f} (ham tahmin: {rf_raw:.2f})")
        print(f"  ⮑ Kelime Puanlama katkısı: {kelime_skor:.2f}")


def ana_menu():
    while True:
        print("\n--- Başvuru Puanlayıcı (0–10 Hibrit Skor) ---")
        print("1. Tek Başvuru Değerlendir")
        print("2. Toplu Değerlendirme (10 örnek)")
        print("3. Çıkış")

        secim = input("Seçiminizi yapın (1-3): ").strip()
        if secim == "1":
            metin = input("Başvuru metnini girin: ").strip()
            if len(metin.split()) < 3:
                print("Uyarı: Metin çok kısa olabilir.")
            hybrid_tahmin_yap(metin)
        elif secim == "2":
            toplu_degerlendirme()
        elif secim == "3":
            print("Çıkılıyor...")
            break
        else:
            print("Geçersiz seçim. 1-3 arası girin.")

if __name__ == "__main__":
    ana_menu()
