import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

class State:
    def __init__(self, names, rec_model):
        self.names = names
        self.rec_model = rec_model

class bookrec:
    def __init__(self, mid_genre):
        for file in os.listdir('data'):
            if mid_genre in file:
                df = pd.read_csv(f'data/{file}')
                break

        df['판매가'] = df['판매가'].str.replace("원", "").str.replace(",", "").astype(int)
        df['판매가 점수'] = 1 - (df['판매가'] - df['판매가'].min()) / (df['판매가'].max() - df['판매가'].min())
        df['중위 장르'] = df['관리분류'].apply(lambda x: x.split('-')[1])

        self.tf = TfidfVectorizer(stop_words='english')
        df['설명'] = df['설명'].fillna('')
        tfidf_matrix = self.tf.fit_transform(df['설명'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(df.index, index=df['상품명']).drop_duplicates()
        self.df = df
        self.names = df['상품명'].tolist()

    def book_recommend(self, selected_books):
        df = self.df
        df['선호도 점수'] = 0
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
    
    #정확성 문제 발생 수정해야함