import pandas as pd
from app.model import BookRecommender
from app.data import load_dataframe

if __name__ == "__main__":
    df = load_dataframe()  # 데이터 로드
    recommender = BookRecommender(df)  # book_recommender 객체 생성

    recommender.select_genre()  # 사용자가 선호하는 장르 선택
    recommender.select_books()  # 사용자가 선호하는 책 선택
    recommender.recommend_books()  # 책 추천