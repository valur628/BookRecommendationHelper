import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
import random

# CSV 파일 읽기
df = pd.read_csv('books.csv')

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
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) 

indices = pd.Series(df.index, index=df['상품명']).drop_duplicates()

def recommend_books(title, cosine_sim=cosine_sim):
    # 해당 책에 다른 책들과의 유사도 계산
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) 
    sim_scores = sim_scores[1:6] # 가장 유사한 상위 5개 추출
    book_indices = [i[0] for i in sim_scores]
    return df['상품명'].iloc[book_indices] 

print(df['관리분류'].unique())
user_genre = input("적합한 장르 이름을 선택해 주세요: ")

# 사용자가 선택한 장르를 기반으로 가장 높은 선호도 점수를 가진 책 5권 찾기
top_books = df[df['관리분류'] == user_genre].nlargest(5, '선호도 점수')['상품명'].tolist()
print('장르에 따른 Top 5 책: ', top_books)
random.shuffle(top_books)
selected_books = input("선호하는 책을 선택하세요: ").split(",")

for book in selected_books:
    recommended_books = recommend_books(book.strip()) 
    # 사용자가 선택한 책과 유사한 책들을 찾고, 해당 책들의 선호도 점수를 높임
    for rec_book in recommended_books:
        try:
           df.loc[df['상품명'] == rec_book, '선호도 점수'] += 0.05 
        except:
            continue

# 최종적으로 가장 높은 선호도 점수를 가진 책 5권 추천
recommendations = df.nlargest(5, '선호도 점수')
print("\n추천된 책 목록: ", recommendations['상품명'].tolist())