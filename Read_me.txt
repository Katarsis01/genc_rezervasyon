KYK Başvuru Öncelik Skoru Tahmin Sistemi
Bu proje, KYK (Kredi ve Yurtlar Kurumu) başvuru metinlerini analiz ederek başvurunun öncelik skorunu (0-10 arası) tahmin eder.
Tahmin, hem makine öğrenmesi modeli (Random Forest) hem de kelime puanlandırma sistemiyle hibrit bir yapıda yapılır.

Projenin Bileşenleri
Bileşen	Açıklama
RandomForestRegressor	Başvuru metnine göre öğrenilmiş skor tahmini yapar.
#KYK Başvuru Öncelik Skoru Tahmin Sistemi

Bu proje, KYK (Kredi ve Yurtlar Kurumu) başvuru metinlerini analiz ederek başvurulara 0 ile 10 arasında bir öncelik skoru tahmin etmeyi amaçlamaktadır. Skor tahmini, makine öğrenmesi (Random Forest) 
ve kelime puanlandırma sistemiyle birlikte çalışan **hibrit bir model** üzerinden yapılmaktadır.

##Temel Özellikler

- Random Forest ile metinlere dayalı skor tahmini (%40 katkı)
- Kelime puanlarına dayalı metin analizi (%60 katkı)
- Gelişmiş Türkçe durak kelime (stopword) filtresi
- Anlamlı kelimeler için %80 benzerlik oranına dayalı gruplayıcı
- 0-10 arasında normalize edilmiş çıktı puanı
- Terminal üzerinden kullanıcı dostu bir tahmin menüsü

##Model Bileşenleri

- **Makine Öğrenmesi**: `RandomForestRegressor` modeli, TF-IDF özellikleri üzerinden başvuru metnini analiz ederek bir skor tahmini yapar.
- **Kelime Puanlandırma**: `kelime_puanlari.txt` dosyasında yer alan anlamlı kelimeler üzerinden metindeki benzer kelimeler puanlandırılır. Bu puanlar normalize edilerek skora katkı sağlar.
- **Hibrit Karar Mekanizması**: Son skor = (Random Forest skoru × 0.4) + (Kelime bazlı skor × 0.6)

##Dosya Yapısı

proje/
├── karar.py # Ana karar mekanizması (çalıştırılabilir arayüz)
├── model_train.py # Random Forest model eğitim dosyası
├── kelime_puan_hesaplayici.py # Kelime bazlı puanları üretir ve kaydeder
├── kelime_puanlari.txt # Anlamlı kelimelerin puanları (önceden çıkarılmış)
├── kyk_rezervasyon_model.pkl # Eğitilmiş Random Forest modeli
├── kyk_rezervasyon_vectorizer.pkl # TF-IDF vektörizer dosyası
├── basvuru_train.csv # Eğitim verisi (başvuru metni + skor)

## Kurulum
pip install pandas scikit-learn matplotlib numpy joblib

Kullanım
Gerekirse modeli ve kelime puanlarını oluştur:
1.Adım:python model_train.py
2.Adım:python kelime_puan_hesaplayici.py
3.Adım:python karar.py (ANA menü burada) 

Program çalıştığında şu menü sunulur:
--- KYK Rezervasyon Modeli ---
1. Tekli Başvuru Tahmini Yap
2. Eğitim Verisinden 10 Başvuru Değerlendir
3. Çıkış
1: Terminale yazdığınız başvuru metnine göre skor tahmini yapar.

2: Eğitim setinden rastgele seçilen 10 cümle üzerinden tahmin vs. gerçek skor karşılaştırması yapılır.
Puanlama Açıklaması
Tahminler 0-10 arasında verilir.

Sonuçlar aşağıdaki şekilde yorumlanır:
0–2.5 → Çok Düşük Öncelik
2.5–5 → Düşük Öncelik
5–7.5 → Orta Öncelik
7.5–9 → Yüksek Öncelik
9–10 → Çok Yüksek Öncelik
Ayrıca her tahminde hangi yüzdelik katkının hangi modelden geldiği de gösterilir.