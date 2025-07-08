# HAPPSYAD: Yüz İfadesi Tanıma Sistemi

Bu proje, görüntü ve kamera akışındaki insan yüzlerinin duygularını tanımak için evrişimli sinir ağlarını (CNN) kullanan bir yüz ifadesi tanıma sistemidir. Proje, FER2013 veri setini kullanarak eğitilmiştir.

## Proje Yapısı
Proje aşağıdaki dosya ve klasörlerden oluşmaktadır:

HAPPYSAD/
├── fer2013_dataset/
│   ├── test/
│   ├── train/
│   └── fer2013.csv
├── face.py
├── model.pth
├── readme.md
└── train.py

  - **`fer2013_dataset/`**: FER2013 duygu tanıma veri setini içerir.
      - `fer2013.csv`: Veri setinin ana CSV dosyası.
      - `test/`, `train/`: Muhtemelen veri setinin resim dosyaları (ancak `train.py` CSV'den piksel verilerini doğrudan okur).
  - **`face.py`**: Eğitilmiş modeli kullanarak görüntü veya kamera akışından yüz ifadelerini algılamak ve tanımak için kullanılan ana uygulamadır.
  - **`model.pth`**: Eğitilmiş CNN modelinin ağırlıklarını içeren dosya. Bu dosya `train.py` çalıştırıldıktan sonra oluşturulur.
  - **`readme.md`**: Bu README dosyası.
  - **`train.py`**: CNN modelini FER2013 veri seti üzerinde eğiten Python betiği.

## Kurulum

Projeyi çalıştırmadan önce gerekli kütüphaneleri kurmanız gerekmektedir.

1.  **Gerekli Kütüphaneleri Kurun:**

    ```bash
    pip install torch torchvision opencv-python-headless pandas scikit-learn pillow matplotlib
    ```

    *Not: Eğer bir grafik arayüzü ile çalışıyorsanız `opencv-python` yerine `opencv-python-headless` kullanabilirsiniz. Genel kullanım için `opencv-python` da uygun olabilir.*

2.  **Veri Setini İndirin:**

    FER2013 veri setini indirmeniz ve `fer2013_dataset` klasörü içine yerleştirmeniz gerekmektedir. `fer2013.csv` dosyasının bu klasörün içinde olması önemlidir. Veri setini Kaggle gibi platformlardan bulabilirsiniz.

## Kullanım

### 1\. Modeli Eğitme (train.py)

Eğer kendi modelinizi eğitmek veya mevcut modeli yeniden eğitmek isterseniz, `train.py` betiğini çalıştırın:

```bash
python train.py
```

Bu betik, FER2013 veri setini yükleyecek, bir CNN modeli tanımlayacak, modeli eğitecek ve en iyi modeli `model.pth` olarak kaydedecektir. Eğitim süreci, konsola her epoch için kayıp ve doğrulama doğruluğunu yazdıracaktır.

### 2\. Yüz İfadesi Tanıma Uygulaması (face.py)

Eğitilmiş modeli kullanarak yüz ifadesi tanıma uygulamasını çalıştırmak için `face.py` betiğini çalıştırın:

```bash
python face.py
```

Uygulama çalıştırıldığında size iki seçenek sunacaktır:

1.  **Kamera ile gerçek zamanlı:** Bilgisayarınızın kamerasını kullanarak gerçek zamanlı yüz ifadesi tespiti yapar.
2.  **Görsel dosya ile analiz:** Bir görsel dosyası seçerek o görseldeki yüzün ifadesini analiz eder.

Uygulamadan çıkmak için `q` tuşuna basın.

## Model Mimarisi

Kullanılan CNN modeli aşağıdaki katmanları içermektedir:

  - **Evrişim Katmanları (`conv`):**
      - `Conv2d(1, 64, kernel_size=3, padding=1)`
      - `ReLU()`
      - `BatchNorm2d(64)`
      - `MaxPool2d(2)`
      - `Conv2d(64, 128, kernel_size=3, padding=1)`
      - `ReLU()`
      - `BatchNorm2d(128)`
      - `MaxPool2d(2)`
      - `Conv2d(128, 256, kernel_size=3, padding=1)`
      - `ReLU()`
      - `BatchNorm2d(256)`
      - `MaxPool2d(2)`
  - **Tam Bağlantılı Katmanlar (`fc`):**
      - `Flatten()`
      - `Linear(256 * 6 * 6, 256)`
      - `ReLU()`
      - `Dropout(0.5)`
      - `Linear(256, 7)` (7 duygu sınıfı için çıktı)

Model, 7 farklı duyguyu tanımak üzere tasarlanmıştır:

  - 0: Öfke
  - 1: İğrenme
  - 2: Korku
  - 3: Mutlu
  - 4: Üzgün
  - 5: Şaşkın
  - 6: Nötr

## Notlar

  - `face.py` dosyasında `font_path` için `C:/Windows/Fonts/arial.ttf` tanımlanmıştır. Eğer bu font sisteminizde yoksa veya farklı bir işletim sistemi kullanıyorsanız, `ImageFont.load_default()` kullanılacaktır. İsteğe bağlı olarak kendi font yolunuzu belirleyebilirsiniz.
  - Performans, kullanılan donanıma (özellikle GPU) ve veri setinin büyüklüğüne bağlı olacaktır.
  - Yüz tespiti için OpenCV'nin `haarcascade_frontalface_default.xml` dosyası kullanılmaktadır. Bu dosyanın OpenCV kurulumunuzla birlikte geldiğinden emin olun.