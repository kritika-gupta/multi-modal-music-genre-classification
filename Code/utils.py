import numpy as np
import pandas as pd
import tqdm, glob

def get_genre_id(genre_name, genre_list):

    dictionary = dict(zip(genre_list, range(len(genre_list))))
    return dictionary[genre_name]

def load_data_trainCNN():
    # load spectrograms

    spectrograms = []
    # for filename in tqdm.tqdm(glob.glob("/content/drive/MyDrive/multi-modal-music-genre-classification/spectrogram_parts_8k_5genres/spectrograms_8k_part_*.csv")):
    for filename in tqdm.tqdm(glob.glob("/content/drive/MyDrive/MS/Fall22/DeepLearning/Deep Learning Project/multi-modal-music-genre-classification/spectrogram_parts_8k_5genres/spectrograms_8k_part_*.csv")):
        df = pd.read_pickle(filename)
        spectrograms.append(df)

    data = pd.concat(spectrograms)
    del(spectrograms)

    genre_list = list(pd.unique(data['main_genre'].values))

    # load lyrics
    
    df_metadata = pd.read_parquet('/content/drive/MyDrive/MS/Fall22/DeepLearning/Deep Learning Project/multi-modal-music-genre-classification/4mula_metadata.parquet')
    # df_metadata = pd.read_parquet('/content/drive/MyDrive/multi-modal-music-genre-classification/4mula_metadata.parquet')
    id_lyrics = df_metadata[['music_id', 'music_lyrics']]

    # merge spectrogram and lyric data based on music_id

    mergedData = pd.merge(data, id_lyrics, on=['music_id'], how='inner')
    mergedData.head(5)

    mergedData.drop_duplicates(subset=['music_id'], inplace=True)

    mergedData = mergedData[['main_genre', 'melspectrogram', 'music_lyrics']]
    mergedData['music_lyrics'] = mergedData['music_lyrics'].apply(lambda string: string.replace("\\n", ". ").replace("\n", ". ").replace("\'", ""))
    mergedData['main_genre'] = mergedData['main_genre'].apply(lambda string: get_genre_id(string, genre_list))


    mergedData['data_pair'] = list(zip(mergedData.melspectrogram, mergedData.music_lyrics))

    X = mergedData['data_pair'].to_numpy()
    y = mergedData['main_genre'].to_numpy()

    ids = [i for i, (spec, lyr) in enumerate(X) if spec.shape==(128, 431)]
    X = X[ids]
    y = y[ids]
    
    X = np.stack(X)
    y = np.stack(y)
    original_zipped_data = list(zip(X, y))
    print(X.shape, y.shape)

    zipped_data = original_zipped_data 

    del(X)
    del(y)

    return zipped_data, genre_list

def load_data_withaug():
    # load spectrograms

    spectrograms = []
    for filename in tqdm.tqdm(glob.glob("/content/drive/MyDrive/Deep\ Learning\ Project/multi-modal-music-genre-classification/spectrogram_parts_8k_5genres/spectrograms_8k_part_*.csv")):
        df = pd.read_pickle(filename)
        spectrograms.append(df)

    data = pd.concat(spectrograms)
    del(spectrograms)

    genre_list = list(pd.unique(data['main_genre'].values))

    # load lyrics
    df_metadata = pd.read_parquet('/content/drive/MyDrive/Deep\ Learning\ Project/multi-modal-music-genre-classification/4mula_metadata.parquet')
    id_lyrics = df_metadata[['music_id', 'music_lyrics']]

    # merge spectrogram and lyric data based on music_id

    mergedData = pd.merge(data, id_lyrics, on=['music_id'], how='inner')
    mergedData.head(5)

    mergedData.drop_duplicates(subset=['music_id'], inplace=True)

    mergedData = mergedData[['main_genre', 'melspectrogram', 'music_lyrics']]
    mergedData['music_lyrics'] = mergedData['music_lyrics'].apply(lambda string: string.replace("\\n", ". ").replace("\n", ". ").replace("\'", ""))
    mergedData['main_genre'] = mergedData['main_genre'].apply(lambda string: get_genre_id(string, genre_list))


    mergedData['data_pair'] = list(zip(mergedData.melspectrogram, mergedData.music_lyrics))

    X = mergedData['data_pair'].to_numpy()
    y = mergedData['main_genre'].to_numpy()

    # loading the augmented datasets
    D2 = np.load('/content/drive/MyDrive/multi-modal-music-genre-classification/augmented_spectrograms/D2.npy', allow_pickle=True)
    D3 = np.load('/content/drive/MyDrive/multi-modal-music-genre-classification/augmented_spectrograms/D3.npy', allow_pickle=True)

    ids = [i for i, (spec, lyr) in enumerate(X) if spec.shape==(128, 431)]
    X = X[ids]
    y = y[ids]

    D2 = D2[ids[:-1]]
    D3 = D3[ids[:-1]]

    y_aug = y[0:4000]
    D2 = D2[0:4000]
    D3 = D3[0:4000]

    D2_new = []
    D3_new = []

    for i, (spec, lyr) in enumerate(X[0:4000]):
      D2_new.append((D2[i], lyr))
      D3_new.append((D3[i], lyr))
    
    X = np.stack(X)
    y = np.stack(y)
    original_zipped_data = list(zip(X, y))
    print(X.shape, y.shape)

    D2_new = np.stack(D2_new)
    D3_new = np.stack(D3_new)
    zipped_data_D2 = list(zip(D2_new, y_aug))
    zipped_data_D3 = list(zip(D3_new, y_aug))
    print(D2_new.shape, y_aug.shape)
    print(D3_new.shape, y_aug.shape)

    zipped_data = original_zipped_data + zipped_data_D2 + zipped_data_D3 

    del(X)
    del(y)
    del(D2_new)
    del(D3_new)

    return zipped_data, genre_list


def load_data(tokenizer):
    # load spectrograms
    print("loading data!")
    print(tokenizer)
    spectrograms = []
    for filename in tqdm.tqdm(glob.glob("/content/drive/MyDrive/multi-modal-music-genre-classification/spectrogram_parts_8k_5genres/spectrograms_8k_part_*.csv")):
        df = pd.read_pickle(filename)
        spectrograms.append(df)

    data = pd.concat(spectrograms)
    del(spectrograms)

    genre_list = list(pd.unique(data['main_genre'].values))

    # load lyrics
    df_metadata = pd.read_parquet('/content/drive/MyDrive/multi-modal-music-genre-classification/4mula_metadata.parquet')
    id_lyrics = df_metadata[['music_id', 'music_lyrics']]

    # merge spectrogram and lyric data based on music_id

    mergedData = pd.merge(data, id_lyrics, on=['music_id'], how='inner')
    mergedData.head(5)

    mergedData.drop_duplicates(subset=['music_id'], inplace=True)

    mergedData = mergedData[['music_id','main_genre', 'melspectrogram', 'music_lyrics']]

    #### format lyrics
    mergedData['music_lyrics'] = mergedData['music_lyrics'].apply(lambda string: string.replace("\\n", ". ").replace("\n", ". ").replace("\'", ""))
    mergedData['main_genre'] = mergedData['main_genre'].apply(lambda string: get_genre_id(string, genre_list))

    #### tokenize lyrics
    lyrics = mergedData['music_lyrics'].to_list()
    x = tokenizer(lyrics, padding=True, truncation=True, return_tensors="np")
    # mergedData['data_pair'] = list(zip(mergedData.melspectrogram, mergedData.music_lyrics))
    #print(mergedData['data_pair'].head(5))

    specs = mergedData['melspectrogram'].to_numpy()
    input_ids = x['input_ids']
    attention_mask = x['attention_mask']

    y = mergedData['main_genre'].to_numpy()
    music_ids = mergedData['music_id'].to_numpy()

    print(specs.shape, input_ids.shape, attention_mask.shape, y.shape)

    ids = [i for i, spec in enumerate(specs) if spec.shape==(128, 431)]
    specs = specs[ids]
    input_ids = input_ids[ids]
    attention_mask = attention_mask[ids]
    y = y[ids]
    music_ids = music_ids[ids]

    # X = np.stack(X)
    # y = np.stack(y)

    # zipped_data = list(zip(X, y))

    #print(len(zipped_data))

    return specs, input_ids, attention_mask, genre_list, y, music_ids
