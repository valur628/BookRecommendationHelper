import pandas as pd
from settings import DATA_PATH

def load_dataframe(csv_file=DATA_PATH):
    df = pd.read_csv(csv_file)
    df['판매가'] = df['판매가'].str.replace("원", "").str.replace(",", "").astype(int)
    df['판매가 점수'] = 1 - (df['판매가'] - df['판매가'].min()) / (df['판매가'].max() - df['판매가'].min())
    df['순위 점수'] = 1 - (df['순위'] - df['순위'].min()) / (df['순위'].max() - df['순위'].min())
    df['선호도 점수'] = df['판매가 점수'] * 0.1 + df['순위 점수'] * 0.9 
    df['중위 장르'] = df['관리분류'].apply(lambda x: x.split('-')[1])
    df['하위 장르'] = df['관리분류'].apply(lambda x: x.split('-')[2] if len(x.split('-')) > 2 else x.split('-')[1])
    return df