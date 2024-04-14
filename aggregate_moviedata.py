import pandas as pd
import os

# Define paths to your dataset files
data_dir = 'ml-1m'
movies_dat_path = os.path.join(data_dir, 'movies.dat')
genome_path = os.path.join(data_dir, 'genome.csv')
movies_csv_path = os.path.join(data_dir, 'movies.csv')

# Load datasets
movies_dat = pd.read_csv(movies_dat_path, delimiter='::', header=None, names=['movieId', 'title', 'genres'], engine='python', encoding='ISO-8859-1')
genome = pd.read_csv(genome_path)
movies_csv = pd.read_csv(movies_csv_path)

# Merge movies.dat with movies.csv to include the year
movies_with_year = movies_dat.merge(movies_csv[['movieId', 'year']], on='movieId', how='left')

# Sort genome data by relevance and select top 5 tags per movie
genome_sorted = genome.sort_values(by=['movieId', 'relevance'], ascending=[True, False])
top_tags = genome_sorted.groupby('movieId').head(5).copy()  # Use copy to avoid SettingWithCopyWarning
top_tags['tag_info'] = top_tags.apply(lambda x: f"{x['tagId']}::{x['tag']}::{x['relevance']:.4f}", axis=1)

# Create compact tag information
compact_top_tags = top_tags.groupby('movieId')['tag_info'].apply('|'.join).reset_index()

# Merge the top tags information with movies data
final_movies = movies_with_year.merge(compact_top_tags, on='movieId', how='left')

# Save the merged data to a new CSV file using a single character delimiter
output_path = os.path.join(data_dir, 'final_merged_movies.csv')
final_movies.to_csv(output_path, sep='|', index=False, header=None, encoding='utf-8')  # Use a valid single-character delimiter

# Replace the delimiter '|' with '::' post hoc
with open(output_path, 'r', encoding='utf-8') as file:
    filedata = file.read()
filedata = filedata.replace('|', '::')

with open(output_path, 'w', encoding='utf-8') as file:
    file.write(filedata)

print("Final merged data saved to:", output_path)
print(final_movies.head())
