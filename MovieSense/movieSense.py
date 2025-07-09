import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#Burada öğrenme amaçlı denemeler yaptım.

data_credits= pd.read_csv("tmdb_5000_credits.csv")
data_movies = pd.read_csv("tmdb_5000_movies.csv")
print(data_credits.head(3))
print(data_movies.head(3))

#Verilerimizin kaç satır kaç sütundan oluştuğunu görelim
print("Data Credits:", data_credits.shape)
print("Data Movies:", data_movies.shape)

print("Data Credits Columns:", data_credits.columns)
print("Data Movies Columns:", data_movies.columns)
#İki dosyayı birleştirebilmemiz için aynı sütun isimlerine sahip olmaları gerekiyor. 
#O yüzden "Movies" verimizin "id" kolonunu "movie_id" olarak değiştiriyoruz.

data_movies.rename(columns = {"id":"movie_id"},inplace=True)

#Veri dosyalarımızı "merge" fonksiyonu ile birleştiriyoruz
data_all = data_movies.merge(data_credits,on="movie_id")

#Birleştirilmiş veri setimizin satır ve sütunlarının uzunluklarına bakıyoruz
print("Data All:", data_all.shape)

print(data_all.head(3))


V = data_all['vote_count']
R = data_all['vote_average']
C = data_all['vote_average'].mean()
m = data_all['vote_count'].quantile(0.75)

#Yeni bir sütun tanımlayarak hesapladığımız değerleri buraya atıyoruz
data_all['weighted_average'] = (V/(V+m) * R) + (m/(m+V) * C)

#Burada tüm verilerimizi azalan şekilde sıralıyoruz(Hesapladığımız Ağırlıklı Puan sütunu üzerinden)
#Sıralama yaptırdığımız yeni veri setimizi belli sütunları göstererek listeliyoruz
movies_ranked = data_all.sort_values('weighted_average', ascending=False)
#print(movies_ranked[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(20))

plt.figure(figsize=(16,5))
ax = sns.barplot(x=movies_ranked['weighted_average'].head(10), y=movies_ranked['original_title'].head(10))

plt.xlim(7,9)
plt.title('Oylarla Seçilen En iyi Filmler', weight='bold')
plt.xlabel('Alınan Ortalama Skor', weight='bold')
plt.ylabel('Film Adı', weight='bold')

plt.savefig('best_movies.png')


popular = movies_ranked.sort_values('popularity', ascending=False)

plt.figure(figsize=(16,5))

ax = sns.barplot(x=popular['popularity'].head(10), y=popular['original_title'].head(10))

plt.title('Oylarla Belirlenen En Popüler Filmler', weight='bold')
plt.xlabel('Popülarite Skoru', weight='bold')
plt.ylabel('Film Adı', weight='bold')

plt.savefig('popular_movies.png')

#Film hakkında önizleme yazılarını listeliyoruz
#print(data_all["overview"].head())

#önizleme yazıları içerisinde boş değer olup olmadığına bakıyoruz
print(data_all.overview.isnull().any())
#Kaç adet boş satır olduğunu buluyoruz
print("Boş Satır Sayısı:", data_all.overview.isnull().sum())

#Veri girilmemiş alanlara boş atama yapıyoruz
data_all['overview'] = data_all['overview'].fillna('')

#TfIdfVectorizer fonksiyonunu scikit-learn kütüphanesinden ekliyoruz
from sklearn.feature_extraction.text import TfidfVectorizer

#Burada ingilizce "the" , "a" ,"an" gibi kelimeleri temizliyoruz.
tfidf = TfidfVectorizer(stop_words='english')

#Fonksiyonu kullanarak dönüşüm işlemini yapıyoruz
tfidf_matrix = tfidf.fit_transform(data_all['overview'])

#Veri setimizin satır ve sütun sayılarını görüntülüyoruz
tfidf_matrix.shape

print(data_all['overview'])


#linear_kernel ekliyoruz
from sklearn.metrics.pairwise import linear_kernel

#cosine benzerliğini hesaplatıyoruz
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#indexlerin ve film başlıklarının ters haritasını oluşturuyoruz. Tekrar eden değerleri listeden çıkarıyoruz
indices = pd.Series(data_all.index, index=data_all['original_title']).drop_duplicates()

# Bu fonksiyon ile fonksiyona göndermiş olduğumuz filme en yakın benzerlikte ki 10 filmi geri döndürecek
def get_recommendations(title, cosine_sim=cosine_sim):
    # Girilen filmin indeksini al
    idx = indices[title]

    # Bu filmin diğer tüm filmlerle olan benzerlik puanlarını listele
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Filmleri benzerlik puanlarına göre sırala
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # En çok benzeyen 10 filmin skorunu al
    sim_scores = sim_scores[1:11]

    # Film indekslerini al
    movie_indices = [i[0] for i in sim_scores]

    # En çok benzeyen 10 filmi döndür
    return data_all['original_title'].iloc[movie_indices]

#The Dark Knight içeriğine benzer 10 film önerisi
print(get_recommendations('The Dark Knight Rises'))

