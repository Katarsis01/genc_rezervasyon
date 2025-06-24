import pandas as pd
import re
from collections import defaultdict
from difflib import SequenceMatcher

# Stopword setini geniş tutabilirsin:
turkish_stopwords = set([
    "ve", "bir", "bu", "ile", "de", "da", "için", "mi", "ne", "ya", "ama",
    "çok", "daha", "en", "gibi", "ise", "veya", "şu", "o", "ki",
    "ben", "sen", "biz", "siz", "onlar", "olarak", "yani", "ancak", "hem",
    "çünkü", "fakat", "her", "hiç", "nasıl", "neden", "niçin", "hangi", "kim", "yok", "var", "gerekiyor", "olabilir", "miyim" , "etmek", "gün" ,"gün","istiyorum", "içi","kalmak"
])

def temizle_ve_parcala(cumle):
    cumle = cumle.lower()
    kelimeler = re.findall(r'\b\w+\b', cumle)
    anlamli = [k for k in kelimeler if k not in turkish_stopwords and len(k) > 2]
    return anlamli

def benzer_mi(kelime1, kelime2, thresh=0.7):
    if kelime1[0] != kelime2[0]:
        return False
    oran = SequenceMatcher(None, kelime1, kelime2).ratio()
    return oran >= thresh

def kelimeyi_bul_ve_grupla(kelime, kelime_listesi):
    for var_kelime in kelime_listesi:
        if benzer_mi(var_kelime, kelime):
            return var_kelime
    return kelime

def kelime_puan_hesapla(df):
    kelime_puanlari = dict()

    for idx, row in df.iterrows():
        cumle = row["başvuru_metni"]
        skor = row["öncelik_skoru"]
        anlamli_kel = temizle_ve_parcala(cumle)

        if not anlamli_kel:
            continue

        kelime_sayisi = len(anlamli_kel)

        for kel in anlamli_kel:
            grup_kelime = kelimeyi_bul_ve_grupla(kel, kelime_puanlari.keys())
            puan = skor / kelime_sayisi

            if grup_kelime in kelime_puanlari:
                eski = kelime_puanlari[grup_kelime]
                yeni = puan
                # Eski ve yeni puanın ortalaması + ufak bir katsayı
                guncel = ((eski + yeni) / 2) * 1.3
                kelime_puanlari[grup_kelime] = guncel
            else:
                kelime_puanlari[grup_kelime] = puan

    return kelime_puanlari

def kaydet(puanlar, dosya="kelime_puanlari.txt"):
    with open(dosya, "w", encoding="utf-8") as f:
        for kelime, puan in sorted(puanlar.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{kelime},{puan:.4f}\n")
    print(f"{dosya} dosyasına başarıyla kaydedildi.")

if __name__ == "__main__":
    df = pd.read_csv("basvuru_train.csv")
    df.dropna(subset=["başvuru_metni", "öncelik_skoru"], inplace=True)

    kelime_puanlari = kelime_puan_hesapla(df)
    kaydet(kelime_puanlari)
