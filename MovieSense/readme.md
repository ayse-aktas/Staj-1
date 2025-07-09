# 🎬 MovieSense: İçerik Tabanlı Film Öneri Sistemi

MovieSense, bir kullanıcının seçtiği filme benzer filmleri önermek amacıyla içerik tabanlı çalışan bir **film öneri sistemi**dir. Sistem, filmlerin açıklamaları üzerinden **TF-IDF (Term Frequency-Inverse Document Frequency)** ve **cosine benzerlik** yöntemiyle çalışır. Ayrıca TMDB API kullanılarak önerilen filmlerin afişleri ve detayları görsel olarak sunulur.

---

## 📦 Proje Yapısı

MovieSense/
├── app.py # Flask uygulamasının ana dosyası
├── tmdb_5000_movies.csv # Film verisi (TMDB veri kümesi)
├── tmdb_5000_credits.csv # Oyuncu/kadro verisi
├── static/
│ └── style.css # CSS stil dosyası (karanlık mod dahil)
├── templates/
│ ├── index.html # Ana sayfa (film listesi + arama)
│ ├── detail.html # Film detay ve öneri sayfası
├── README.md # Proje açıklama dosyası

## 🚀 Özellikler

- 🔍 **Akıllı Arama (Autocomplete)**: Film adını yazarken öneriler otomatik görünür.
- 🧠 **İçerik Tabanlı Film Önerisi**: Seçilen filmin açıklamasına benzer filmleri önerir.
- 🎨 **Modern Arayüz & Karanlık Mod**: Hafif ve kullanıcı dostu tema, gece modu seçeneği ile.
- 🎬 **Film Detay Sayfası**: Seçilen film hakkında açıklama, puan, yayın tarihi ve benzer filmler sunar.
- 🎞️ **Afişlerle Listeleme**: Tüm filmler posterleriyle gösterilir.
- 🧩 **Tür Filtreleme (genre)**: Filmler belirli türlere göre listelenebilir.

## 🧠 Kullanılan Teknolojiler

- **Python**
- **Flask** (Backend ve routing)
- **scikit-learn** (`TfidfVectorizer`, `linear_kernel`)
- **TMDB API** (Poster verisi ve film bilgileri)
- **HTML / CSS / JavaScript** (Önyüz)
- **Pandas** (Veri analizi)

## ⚙️ Kurulum

### 1. Gerekli Kütüphaneleri Kur

```bash
pip install flask pandas scikit-learn requests
```

2. TMDB API Anahtarını Al ve Ekleyin
https://www.themoviedb.org/settings/api adresinden bir API anahtarı al.

app.py dosyasında TMDB_API_KEY değişkenine kendi anahtarını gir:
TMDB_API_KEY = "your_tmdb_api_key"

3. Uygulamayı Başlat
```bash
python app.py
```

Tarayıcıdan şunu ziyaret edin:
http://127.0.0.1:5000/

🎥 Kullanım
Ana Sayfa: Tüm filmleri afişleriyle gör. Dilersen tür filtresi seç ya da film adını ara.
Film Seçimi: Bir filme tıkla.
Detay Sayfası: Film bilgisi + benzer 10 öneri gösterilir.
Karanlık Mod: Sayfanın sağ üstündeki 🌙 tuşuna tıklayarak tema değiştirilebilir.

📚 Kaynaklar
TMDB 5000 Movie Dataset
The Movie Database API
scikit-learn: TF-IDF