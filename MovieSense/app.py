from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

TMDB_API_KEY = "f4beed1315f02e39e7445838906b04a4"

def get_poster_url(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results")
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return ""

# Load data
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')
movies.rename(columns={"id": "movie_id"}, inplace=True)
data = movies.merge(credits, on="movie_id")
data['overview'] = data['overview'].fillna('')
data['genres'] = data['genres'].fillna('[]')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data['original_title']).drop_duplicates()

# Movie recommendation based on description similarity
def get_recommendations(title):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    results = []
    for i in movie_indices:
        movie_title = data.iloc[i]['original_title']
        poster_url = get_poster_url(movie_title)
        results.append({
            'title': movie_title,
            'poster': poster_url,
            'overview': data.iloc[i]['overview'],
            'release_date': data.iloc[i]['release_date'],
            'vote_average': data.iloc[i]['vote_average']
        })
    return results

@app.route('/')
def index():
    selected_genre = request.args.get('genre')
    sorted_data = data.sort_values(by='popularity', ascending=False)
    top_movies = sorted_data.head(100)
    film_listesi = []
    for _, row in top_movies.iterrows():
        title = row['original_title']
        poster = get_poster_url(title)
        if selected_genre:
            if selected_genre not in row['genres']:
                continue
        film_listesi.append({'title': title, 'poster': poster})
    genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Fantasy', 'Horror', 'Romance', 'Sci-Fi']
    return render_template('index.html', filmler=film_listesi, genres=genres, selected_genre=selected_genre)

@app.route('/movie/<title>')
def movie_detail(title):
    movie = data[data['original_title'] == title]
    if movie.empty:
        return f"{title} bulunamadı", 404
    movie = movie.iloc[0]
    poster = get_poster_url(title)
    overview = movie['overview']
    release_date = movie['release_date']
    vote_average = movie['vote_average']
    similar_movies = get_recommendations(title)
    return render_template('detail.html', title=title, poster=poster, overview=overview,
                           release_date=release_date, vote_average=vote_average,
                           similar_movies=similar_movies)

@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('q', '').lower()
    matched = [title for title in data['original_title'] if title.lower().startswith(query)]
    return jsonify(matched[:10])

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return render_template('index.html', filmler=[], genres=[], selected_genre=None)
    matched_movies = data[data['original_title'].str.contains(query, case=False, na=False)]
    film_listesi = []
    for _, row in matched_movies.iterrows():
        film_listesi.append({
            'title': row['original_title'],
            'poster': get_poster_url(row['original_title'])
        })
    genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Fantasy', 'Horror', 'Romance', 'Sci-Fi']
    return render_template('index.html', filmler=film_listesi, genres=genres, selected_genre=None)

@app.context_processor
def inject_theme():
    return dict(dark_mode=request.cookies.get('theme') == 'dark')

if __name__ == '__main__':
    app.run(debug=True)
