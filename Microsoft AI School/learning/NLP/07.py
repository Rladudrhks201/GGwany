# 간단한 추천 시스템 생성
import numpy as np
import pandas as pd
from math import sqrt

# 데이터 읽기
movies = pd.read_csv('./movies.csv')
ratings = pd.read_csv('./ratings.csv')

print(ratings.head(2))

# 테이블 조인
data = pd.merge(ratings, movies, on="movieId")
column = ['userId', 'movieId', 'rating', 'title', 'genres']
data = data[column]
print(data.head(2))

moviedata = data.pivot_table(index='movieId', columns='userId')['rating']

# NaN 값을 -1로 변경
moviedata.fillna(-1, inplace=True)


# kdd 유사도 함수
def sim_distance(data, n1, n2):
    sum = 0
    # 두 사용자가 모두 본 영화를 기준
    for i in data.loc[n1, data.loc[n1, :] >= 0].index:
        if data.loc[n2, i] >= 0:
            # 누적합
            sum += pow(data.loc[n1, i] - data.loc[n2, i], 2)
    return sqrt(1 / (sum + 1))  # 유사도 형식으로 출력


def top_match(data, name, rank=5, simf=sim_distance):
    # 나와 유사도가 높은 유저 매칭
    simList = []
    for i in data.index[-10:]:
        # 전체 범위로하면 실행시간이 너무 오래걸림
        if name != i:
            simList.append((simf(data, name, i), i))
        simList.sort()
        simList.reverse()
    return simList[:rank]


def recommendation(data, person, simf=sim_distance):
    # 추천 시스템 함수
    result_top = top_match(data, person, len(data))
    score_dic = {}
    sim_dic = {}
    my_list = []
    for sim, name in result_top:
        if sim < 0:
            continue
        for movie in data.loc[person, data.loc[person, :] < 0].index:
            sim_sum = 0
            if data.loc[name, movie] >= 0:
                sim_sum += data.loc[name, movie]

                score_dic.setdefault(movie, 0)
                score_dic[movie] += sim_sum

                sim_dic.setdefault(movie, 0)
                sim_dic[movie] += sim

    for key in score_dic:
        my_list.append((score_dic[key] / sim_dic[key], key))
    my_list.sort()
    my_list.reverse()
    return my_list


# 20번 user가 안본 영화 중에서 추천 점수가 가장 높은 순으로 예상 평점과 영화 제목을 추천 (15개까지)
movie_list = []
for rate, m_id in recommendation(moviedata, 20):
    movie_list.append((rate, movies.loc[movies['movieId'] == m_id, 'title'].values[0]))
    if len(movie_list) == 15:
        break

print(movie_list)
