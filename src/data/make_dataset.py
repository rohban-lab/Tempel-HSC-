import pandas as pd
import random
import math
import os


def read_trigram_vecs(data_path='./data/raw/'):
    """
  Reads the csv file containing 100 dimensional prot vecs, the
  data_path argument indicating where it is located.
  Returns a dictionary that maps a 3gram of amino acids to its
  index and a numpy array containing the trigram vecs.
  """

    prot_vec_file = 'protVec_100d_3grams.csv'

    df = pd.read_csv(data_path + prot_vec_file, delimiter='\t')
    trigrams = list(df['words'])
    trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
    trigram_vecs = df.loc[:, df.columns != 'words'].values

    return trigram_to_idx, trigram_vecs


def read_strains_from(data_files, data_path):
    """
  Reads the raw strains from the data_files located by the data_path.
  Returns a pandas series for each data file, contained in a ordered list.
  """
    # _, data_path = subtype_selection(subtype)
    raw_strains = []
    for file_name in data_files:
        df = pd.read_csv(data_path + file_name)
        strains = df['seq']
        raw_strains.append(strains)

    return raw_strains


def train_test_split_strains(strains_by_year, test_split, cluster):
    """
  Shuffles the strains in each year and splits them into two disjoint sets,
  of size indicated by the test_split.
  Expects and returns pandas dataframe or series.
  """
    train_strains, test_strains = [], []
    if cluster == 'random':
        for strains in strains_by_year:
            num_of_training_examples = int(math.floor(strains.count() * (1 - test_split)))
            shuffled_strains = strains.sample(frac=1).reset_index(drop=True)
            train = shuffled_strains.iloc[:num_of_training_examples].reset_index(drop=True)
            test = shuffled_strains.iloc[num_of_training_examples:].reset_index(drop=True)
            train_strains.append(train)
            test_strains.append(test)
    else:
        # change the starting index for the time-series training samples for multiple experiments
        for strains in strains_by_year:
            num_of_training_examples = int(math.floor(strains.count() * (1 - test_split)))
            train = strains.iloc[:800].reset_index(drop=True)
            test = strains.iloc[800:1000].reset_index(drop=True)
            train_strains.append(train)
            test_strains.append(test)
    return train_strains, test_strains


def read_clusters_from(data_files, start_clusters=[0], no_clusters=1, method='DBSCAN', data_path='../data/interim/'):
    """Reads in data and picks linked clusters"""
    raw_strains = []
    clusters_to_pick = start_clusters
    for file_name in data_files:
        df = pd.read_csv(data_path + method + '.' + file_name)
        df = df[df.cluster.isin(clusters_to_pick)]

        next_year_clusters = []
        for cluster in clusters_to_pick:
            string_list = df[df['cluster'] == cluster]['links'].iloc[0][1:-1].split(' ')
            next_year_clusters += [int(i) for i in string_list]
        clusters_to_pick = next_year_clusters[:no_clusters]

        uncertain_strains = df['seq']
        sequences = replace_uncertain_AAs(uncertain_strains)
        raw_strains.append(sequences)

    return raw_strains
