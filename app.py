import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# CSV 파일 읽기
df = pd.read_csv('DB/books.csv')

# 판매가에서 '원' 제거 및 숫자로 변환
df['판매가'] = df['판매가'].str.replace("원", "").str.replace(",", "").astype(int)
df['판매가 점수'] = 1 - (df['판매가'] - df['판매가'].min()) / (df['판매가'].max() - df['판매가'].min())
df['순위 점수'] = 1 - (df['순위'] - df['순위'].min()) / (df['순위'].max() - df['순위'].min())
df['선호도 점수'] = df['판매가 점수'] * 0.1 + df['순위 점수'] * 0.9 

# 중위 장르와 하위 장르 분리
df['중위 장르'] = df['관리분류'].apply(lambda x: x.split('-')[1])
df['하위 장르'] = df['관리분류'].apply(lambda x: x.split('-')[2] if len(x.split('-')) > 2 else x.split('-')[1])

tfidf = TfidfVectorizer(stop_words='english')
df['설명'] = df['설명'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['설명'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['상품명']).drop_duplicates()

genres = df['하위 장르'].unique().tolist()
for i, genre in enumerate(genres, 1):
    print(f"{i}: {genre}")
genre_idx = int(input("번호로 선호장르를 선택해 주세요: "))
user_genre = genres[genre_idx-1]

genre_df = df[df['중위 장르'].eq(user_genre) | df['하위 장르'].eq(user_genre)]
assert genre_df.empty is False, "선택한 장르에 해당하는 책이 없습니다."
selected_books = genre_df.sample(n=10)
for i, book in selected_books.iterrows():
    print(f"{i}: {book['상품명']}")
selected_books_idx = input("번호로 선호하는 책을 골라주세요(여러 개 선택 가능하며, 쉼표로 구분): ").split(",")
selected_books = [selected_books.loc[int(no), '상품명'] for no in selected_books_idx]

df.loc[df['중위 장르'].eq(user_genre), '선호도 점수'] += 1.0
df.loc[df['하위 장르'].eq(user_genre), '선호도 점수'] += 0.5

for book in selected_books:
    idx = indices[book]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1] if not isinstance(x[1], np.ndarray) else x[1][0], reverse=True)
    sim_scores = sim_scores[1:31]
    book_indices = [i[0] for i in sim_scores]
    for i in book_indices:
        df.loc[i, '선호도 점수'] += df.loc[i, '선호도 점수'] * (sim_scores[book_indices.index(i)][1] if not isinstance(sim_scores[book_indices.index(i)][1], np.ndarray) else sim_scores[book_indices.index(i)][1][0])

recommendations = df.nlargest(5, '선호도 점수')
print("추천된 책들:\n", recommendations[['상품명', '선호도 점수']])