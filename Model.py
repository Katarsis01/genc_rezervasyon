import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Kaydetme yolları
MODEL_PATH = "kyk_rezervasyon_model.pkl"
VECTORIZER_PATH = "kyk_rezervasyon_vectorizer.pkl"

turkish_stopwords = [
    "ve", "bir", "bu", "ile", "de", "da", "için", "mi", "ne", "ya", "ama",
    "çok", "daha", "en", "gibi", "ise", "ile", "veya", "şu", "bu", "o", "ki",
    "ben", "sen", "biz", "siz", "onlar", "olarak", "yani", "ancak", "hem",
    "çünkü", "fakat", "her", "hiç", "nasıl", "neden", "niçin", "hangi", "kim"
]

# Model ve vectorizer yükleme
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model dosyaları bulunamadı. Lütfen önce modeli eğitin.")
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# Model eğitme
def train_model():
    df1 = pd.read_csv("basvuru_train.csv")
    df = pd.concat([df1], ignore_index=True)

    df.dropna(subset=["başvuru_metni", "öncelik_skoru"], inplace=True)

    X = df["başvuru_metni"]
    y = df["öncelik_skoru"]

    # TF-IDF iyileştirmeleri
    vectorizer = TfidfVectorizer(stop_words=turkish_stopwords, ngram_range=(1, 2), min_df=2)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Performans metrikleri
    predictions = model.predict(X_test)
    print("\n--- Model Performansı ---")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("MSE:", mean_squared_error(y_test, predictions))
    print("R2 Skoru:", r2_score(y_test, predictions))
    # Tahmin vs Gerçek grafik
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.title("Model Performansı: Gerçek vs Tahmin")
    plt.grid(True)
    plt.show()
    # Skor dağılımı
    plt.figure(figsize=(8, 4))
    plt.hist(y, bins=10, edgecolor='black')
    plt.title("Eğitim Verisi Skor Dağılımı")
    plt.xlabel("Öncelik Skoru")
    plt.ylabel("Frekans")
    plt.grid(True)
    plt.show()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

# Tahmin ve yorumlama fonksiyonu
def predict_application(text):
    model, vectorizer = load_model()
    if model is None:
        return
    # Metin temizleme
    if ',' in text:
        text = text.split(',')[0]
    text = text.strip()
    # Çok kısa metin uyarısı
    if len(text.split()) < 3:
        print("Uyarı: Metin çok kısa, sonuç güvenilir olmayabilir.")
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    # Yorumlama
    if prediction > 8:
        yorum = "Yüksek Öncelik"
    elif prediction > 5:
        yorum = "Orta Öncelik"
    else:
        yorum = "Düşük Öncelik"

    print(f"\n Model Tahmini Öncelik Skoru: {prediction:.2f} / 10 → {yorum}")

# Ana Menü
def main_menu():
    while True:
        print("\n--- KYK Rezervasyon Modeli ---")
        print("1. Modeli Eğit")
        print("2. Başvuru Tahmini Yap")
        print("3. Çıkış")

        choice = input("Seçiminizi yapın (1-3): ")

        if choice == "1":
            train_model()
        elif choice == "2":
            text = input("Başvuru metnini girin: ")
            predict_application(text)
        elif choice == "3":
            print("Programdan çıkılıyor...")
            break
        else:
            print("Geçersiz seçim. Lütfen 1-3 arasında bir değer girin.")

# Program başlat
if __name__ == "__main__":
    main_menu()
