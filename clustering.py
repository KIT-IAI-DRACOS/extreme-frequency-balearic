import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import seaborn as sns

def get_features(generation_data, events):
    event_list = events.copy()
    
    for data in ['tnr', 'trn', 'mallorca-ibiza_link', 'ibiza-formentera_link', 'mallorca-menorca_link', 'demand_programmed', 'demand_forecast']:
            if data in generation_data.columns:
                generation_data = generation_data.drop(data, axis=1)

    for gen_type in generation_data.columns:
            events[gen_type] = events['Start'].map(generation_data[gen_type])

    features = calculate_ramp_rates(event_list, generation_data)
    return features


def calculate_ramp_rates(features, generation_data):
    generation_data = generation_data.reset_index()
    
    # Calculate the diff for each generation column
    for column in generation_data.columns:
        if column.startswith('generation_') or column.startswith('balearic') or column.startswith('demand_real'):
            generation_data['ramp_' + column] = generation_data[column].diff()

    # Merge and extract relevant data for each event
    for index, event in features.iterrows():
        start, end = event['Start'], event['End']
        event_data = generation_data[(generation_data['time'] >= start) & (generation_data['time'] <= end)]

        # Find max or min ramp rates
        for column in generation_data.columns:
            if column.startswith('ramp'):
                max_abs_ramp_idx = event_data[column].abs().argmax() # Absolute max value
                features.at[index, column] = event_data[column].iloc[max_abs_ramp_idx]

    return features
