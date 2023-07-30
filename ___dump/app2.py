import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from random import sample

# 데이터 로드
data = pd.read_csv('your_data.csv')
data = data.sample(n=10000) # 1만 개 샘플 가져오기

# 속성 변환 및 스케일링
data['판매가'] = data['판매가'].str.replace('원', '').str.replace(',', '').astype('int')
scaler = MaxAbsScaler()
data[['순위', '판매가']] = scaler.fit_transform(data[['순위', '판매가']])

# 선택에 따른 점수 계산
selected_books = [] # 사용자가 선택한 책 리스트
preference = '0.5*순위 + 0.5*판매가' # 선호도 점수 계산 방식

# 10회 반복
for i in range(10):
  # 데이터 샘플링 후 사용자에게 표시
  displayed_books = data.sample(n=5)
  print(displayed_books['상품명'])
  
  # 사용자로부터 선택 받아서 리스트에 저장
  selected_book = input("선택할 책 이름을 입력하세요: ")
  selected_books.append(selected_book)
  
  # 선택한 도서에 가중치 추가
  selected_index = data[data['상품명'] == selected_book].index
  data.loc[selected_index, '선호도 점수'] = data.loc[selected_index].eval(preference) + len(selected_books)

# 선호 장르에 따른 점수 부여
preferred_genre = input('선호하는 장르를 입력하세요: ')
data.loc[data['관리분류'] == preferred_genre, '선호도 점수'] += 1

# Tf-idf 벡터화
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['설명'].dropna())
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 선택한 책과 유사한 책에 가중치 부여
for book in selected_books:
  sim_scores = list(enumerate(cosine_sim[data[data['상품명'] == book].index[0]]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
  for book, score in sim_scores:
    data.loc[book, '선호도 점수'] += score

# 최종 도서 목록 출력
recommendations = data.sort_values('선호도 점수', ascending=False).iloc[:5]['상품명']
print(recommendations)