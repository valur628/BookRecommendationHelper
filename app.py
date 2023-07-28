import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random

# CSV 파일 읽기
df = pd.read_csv('DB/books.csv')

# 판매가에서 "원" 제거 및 숫자로 변환
df['판매가'] = df['판매가'].str.replace("원", "").str.replace(",", "").astype(int)
# 판매 가격 점수를 정규화
df['판매가 점수'] = 1 - (df['판매가'] - df['판매가'].min()) / (df['판매가'].max() - df['판매가'].min())
# 순위 점수를 정규화
df['순위 점수'] = 1 - (df['순위'] - df['순위'].min()) / (df['순위'].max() - df['순위'].min())
# 판매가 점수와 순위 점수를 결합한 선호도 점수를 생성, 직관적으로 가중치 부여
df['선호도 점수'] = df['판매가 점수'] * 0.1 + df['순위 점수'] * 0.9 

# TF-IDF를 이용해서 책의 설명 벡터화, 그 다음 코사인 유사도를 이용해 가장 유사한 책들을 찾음
tfidf = TfidfVectorizer(stop_words='english')
df['설명'] = df['설명'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['설명'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['상품명']).drop_duplicates()

df['설명'] = df['설명'].fillna('')

def recommend_books(df, title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 상위 10개가 아닌 상위 50개를 구하고 그 중 가장 '선호도 점수'가 높은 책 5권을 추천한다.
    sim_scores = sim_scores[1:51]  
    book_indices = [i[0] for i in sim_scores]
    
    # 사용자가 선택한 책과 유사한 책들을 찾고, 해당 책들의 선호도 점수를 높임
    for index in book_indices:
        df.loc[index, '선호도 점수'] += 0.05

# 사용자에게 선호하는 장르 선택 받기  
genres = df['관리분류'].unique().tolist()
for i, genre in enumerate(genres, 1):
    print(f"{i}: {genre}")
genre_no = int(input("번호로 선호장르를 선택해 주세요: "))
user_genre = genres[genre_no-1]

# 사용자가 선택한 장르를 기반으로 랜덤하게 책 5권 추출
genre_df = df[df['관리분류'] == user_genre]
selected_books = genre_df.sample(n=5)
for i, row in selected_books.iterrows():
    print(f"{i}: {row['상품명']}")
selected_books_no = input("번호로 선호하는 책을 골라주세요(여러 개 선택 가능하며, 쉼표로 구분): ").split(",")
selected_books = [selected_books.loc[int(no), '상품명'] for no in selected_books_no]  # idx를 그대로 사용

for book in selected_books:
    recommend_books(df, book.strip())  # 사용자가 선택한 책과 유사한 책들을 찾고, 해당 책들의 선호도 점수를 높임

# 최종적으로 가장 높은 선호도 점수를 가진 책 5권 추천
recommendations = df.nlargest(5, '선호도 점수')
print("추천된 책 목록: ", recommendations[['상품명', '선호도 점수']])