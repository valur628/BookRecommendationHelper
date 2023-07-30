import pandas as pd
from app.data import load_dataframe

class BookRecommender:

    def __init__(self, df):
        self.df = df
        self.user_genre = None
        self.selected_books = None

    def select_genre(self):
        genres = self.df['하위 장르'].unique().tolist()
        for i, genre in enumerate(genres, 1):
            print(f"{i}: {genre}")
        genre_idx = int(input("번호로 선호장르를 선택해 주세요: "))
        self.user_genre = genres[genre_idx-1]

    def select_books(self):
        genre_df = self.df[self.df['중위 장르'].eq(self.user_genre) | self.df['하위 장르'].eq(self.user_genre)]
        assert genre_df.empty is False, "선택한 장르에 해당하는 책이 없습니다."
        selected_books = genre_df.sample(n=10)
        selected_books.reset_index(inplace=True)

        for i, book in selected_books.iterrows():
            print(f"{i+1}: {book['상품명']}")
        selected_books_idx = input("번호로 선호하는 책을 골라주세요(여러 개 선택 가능하며, 쉼표로 구분): ").split(",")
        self.selected_books = [selected_books.loc[int(no)-1, '상품명'] for no in selected_books_idx]

    def recommend_books(self):
        self.df.loc[self.df['중위 장르'].eq(self.user_genre), '선호도 점수'] += 1.0
        self.df.loc[self.df['하위 장르'].eq(self.user_genre), '선호도 점수'] += 0.5
            
        indices = pd.Series(self.df.index, index=self.df['상품명']).drop_duplicates()
        self.df.loc[indices[self.selected_books[0]]:, '선호도 점수'] += 1.0

        recommendations = self.df.nlargest(5, '선호도 점수')
        print("추천된 책들:\n", recommendations[['상품명', '선호도 점수']])