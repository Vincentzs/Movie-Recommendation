#Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import ParameterGrid

from envs import OfflineEnv
from recommender import DRRAgent
import csv
import os
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/ml-1m')
# DATA_DIR = os.path.join(ROOT_DIR, 'ml-25m/ml-25m')
# DATA_DIR = os.path.join(ROOT_DIR, 'ml-latest-small/ml-latest-small')
STATE_SIZE = 10
MAX_EPISODE_NUM = 10

def custom_read_data(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:
        # Skip the header or manage accordingly if there's no header
        next(file)  # Uncomment this line if there's a header
        data = []
        for line in file:
            parts = line.strip().split(':')
            if len(parts) > 5:
                # Rejoin incorrectly split parts for Title or AggregatedTags
                movie_id = parts[0]
                title = ':'.join(parts[1:-3])  # Assumes title is the field being split incorrectly
                genres = parts[-3]
                year = parts[-2]
                aggregated_tags = parts[-1]
                data.append([movie_id, title, genres, year, aggregated_tags])
            else:
                data.append(parts)
    return data

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def grid_search(env, users_num, items_num, param_grid):
    best_precision = -np.inf
    best_config = None
    results = []

    for params in ParameterGrid(param_grid):
        print(f"Testing configuration: {params}")
        agent = DRRAgent(env, users_num, items_num, STATE_SIZE, use_wandb=False, **params)
        agent.actor.build_networks()
        agent.critic.build_networks()
        history = agent.train(MAX_EPISODE_NUM, load_model=False)
        
        # Safely calculate average precision
        # Use 'if history else 0' to handle empty lists or None
        avg_precision = np.mean(history) if history else 0 

        results.append((params, avg_precision))
        if avg_precision > best_precision:
            best_precision = avg_precision
            best_config = params

        print(f"Configuration {params} achieved average precision: {avg_precision}")

    print(f"Best configuration: {best_config} with precision: {best_precision}")
    return best_config, results

if __name__ == "__main__":

    print('Data loading...')
    #Loading datasets
    # Ensure all elements can be converted to integers
    ratings_list = [[int(x) if x.isdigit() else 0 for x in i.strip().split("::")] for i in open(os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
    users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
    # ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = np.uint32)
    ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    ratings_df['UserID'] = ratings_df['UserID'].astype(np.uint32)
    ratings_df['MovieID'] = ratings_df['MovieID'].astype(np.uint32)
    ratings_df['Rating'] = ratings_df['Rating'].astype(np.uint32)
    ratings_df['Timestamp'] = ratings_df['Timestamp'].astype(np.uint32)
    movies_data = custom_read_data(os.path.join(DATA_DIR, 'merged_movies.dat'))
    movies_df = pd.DataFrame(movies_data, columns=['MovieID', 'Title', 'Genres', 'Year', 'AggregatedTags'])

    print("Data loading complete!")
    print("Data preprocessing...")
    print(movies_df.head())

    # 영화 id를 영화 제목으로
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
    # ratings_df = ratings_df.applymap(int)
    ratings_df = ratings_df.astype(int)

    
    # 유저별로 본 영화들 순서대로 정리
    users_dict = np.load('./data/user_dict.npy', allow_pickle=True)

    # 각 유저별 영화 히스토리 길이
    users_history_lens = np.load('./data/users_histroy_len.npy')

    users_num = max(ratings_df["UserID"])+1
    items_num = max(ratings_df["MovieID"])+1

    # Training setting
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k:users_dict.item().get(k) for k in range(1, train_users_num+1)}
    train_users_history_lens = users_history_lens[:train_users_num]

    print('DONE!')
    time.sleep(2)

    env = OfflineEnv(train_users_dict, train_users_history_lens, movies_id_to_movies, STATE_SIZE)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE, use_wandb=False)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.train(MAX_EPISODE_NUM, load_model=False)
    
    # Define hyperparam
    param_grid = {
        # Keep the embedding dim as 100 for now
        'embedding_dim': [100],
        'actor_hidden_dim': [128, 256],
        'actor_learning_rate': [0.1, 0.01, 0.001, 0.0001],
        'critic_hidden_dim': [128, 256],
        'critic_learning_rate': [0.1, 0.01, 0.001, 0.0001],
        'batch_size': [16, 32, 64, 128],
        'load_weights': [True],
    }

    # best_config, results = grid_search(env, users_num, items_num, param_grid)