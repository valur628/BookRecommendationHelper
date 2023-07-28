import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import random

# CSV 파일 읽기
df = pd.read_csv('DB/books.csv')

# 판매가에서 "원" 제거 및 숫자로 변환
df['판매가'] = df['판매가'].str.replace("원", "").str.replace(",", "").astype(int)
# 선호 가격(보통 더 낮은 가격을 선호) 점수를 '판매가 점수'로 정의, 이를 정규화
df['판매가 점수'] = 1 - (df['판매가'] - df['판매가'].min()) / (df['판매가'].max() - df['판매가'].min())

# 선호 순위(순위가 매겨진 책을 선호) 점수를 '순위 점수'로 정의, 이를 정규화
df['순위 점수'] = 1 - (df['순위'] - df['순위'].min()) / (df['순위'].max() - df['순위'].min())

# '판매가 점수'와 '순위 점수'를 결합한 '선호도 점수' 생성, 직관적으로 가중치 부여
df['선호도 점수'] = df['판매가 점수'] * 0.2 + df['순위 점수'] * 0.8 

# Surprise Library를 사용해서, 선호 장르 배우기
reader = Reader(rating_scale=(0, 1)) 
data = Dataset.load_from_df(df[['관리분류', '상품명', '선호도 점수']], reader)
trainset = data.build_full_trainset()
algo = SVD() # Singular Value Decomposition
algo.fit(trainset)

# TF-IDF를 이용해서 책의 설명 벡터화, 그 다음 코사인 유사도를 이용해 가장 유사한 책들을 찾음
tfidf = TfidfVectorizer(stop_words='english')
df['설명'] = df['설명'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['설명'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['상품명']).drop_duplicates()

df['설명'] = df['설명'].fillna('')

def recommend_books(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    # sim_scores의 설정을 통해 리스트의 첫 번째 요소로 설정 
    sim_scores = sorted(sim_scores, key=lambda x: np.max(x[1]), reverse=True) 
    sim_scores = sim_scores[1:6]  # The most similar 5 books
    book_indices = [i[0] for i in sim_scores]
    return df['상품명'].iloc[book_indices]

# 사용자에게 선호하는 장르 선택 받기  
genres = df['관리분류'].unique().tolist()
for i, genre in enumerate(genres, 1):
    print(f"{i}: {genre}")
genre_no = int(input("번호로 선호장르를 선택해 주세요: "))
user_genre = genres[genre_no-1]

#사용자가 선택한 장르를 기반으로 가장 높은 선호도 점수를 가진 책 5권 찾기
top_books = df[df['관리분류'] == user_genre].nlargest(5, '선호도 점수')['상품명'].tolist()
for i, bookname in enumerate(top_books, 1):
    print(f"{i}: {bookname}")

selected_books_no = input("번호로 선호하는 책을 골라주세요(여러 개 선택 가능하며, 쉼표로 구분): ").split(",")
selected_books = [top_books[int(no)-1] for no in selected_books_no]

for book in selected_books:
    recommended_books = recommend_books(book.strip()) 
    # 사용자가 선택한 책과 유사한 책들을 찾고, 해당 책들의 선호도 점수를 높임
    for rec_book in recommended_books:
        try:
           df.loc[df['상품명'] == rec_book, '선호도 점수'] += 0.05 
        except:
            continue

# 최종적으로 가장 높은 선호도 점수를 가진 책 5권 추천
recommendations = df.drop_duplicates('상품명').nlargest(5, '선호도 점수')
print("\n추천된 책 목록: ", recommendations['상품명'].tolist())