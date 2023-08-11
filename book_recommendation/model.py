import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class State:
    def __init__(self, genre, names, mid_genre, rec_model):
        self.genre = genre
        self.names = names
        self.mid_genre = mid_genre
        self.rec_model = rec_model

class bookrec:
    def __init__(self):
        df = pd.read_csv('data/books.csv')
        df['판매가'] = df['판매가'].str.replace("원", "").str.replace(",", "").astype(int)
        df['판매가 점수'] = 1 - (df['판매가'] - df['판매가'].min()) / (df['판매가'].max() - df['판매가'].min())
        df['중위 장르'] = df['관리분류'].apply(lambda x: x.split('-')[1])
        df['하위 장르'] = df['관리분류'].apply(lambda x: x.split('-')[2])
        self.tf = TfidfVectorizer(stop_words='english')
        df['설명'] = df['설명'].fillna('')
        tfidf_matrix = self.tf.fit_transform(df['설명'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(df.index, index=df['상품명']).drop_duplicates()
        self.df = df
        self.genre = df['하위 장르'].unique().tolist()
        self.names = df['상품명'].tolist()

    def book_recommend(self, user_genre, selected_books):
        df = self.df
        df['선호도 점수'] = 0
        df.loc[df['중위 장르'].eq(user_genre) | df['하위 장르'].eq(user_genre), '선호도 점수'] += 0.01
        for book in selected_books:
            idx = self.indices[book]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1] if not isinstance(x[1], np.ndarray) else x[1][0], reverse=True)
            sim_scores = sim_scores[1:31]
            book_indices = [i[0] for i in sim_scores]
            for i in book_indices:
                df.loc[i, '선호도 점수'] += sim_scores[book_indices.index(i)][1] if not isinstance(sim_scores[book_indices.index(i)][1], np.ndarray) else sim_scores[book_indices.index(i)][1][0]

        recommendations = df.nlargest(5, '선호도 점수')
        return [dict(row) for _, row in recommendations.iterrows()]