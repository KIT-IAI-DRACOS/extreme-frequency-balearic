import pandas as pd  
import pickle  
import os  
#from scraper import SpanishScraper  
  
def load_data():  
    # Load frequency data  
    with open('balearic_frequency_cleaned.pkl', 'rb') as file:  
        frequency_data_raw = pickle.load(file)  
    frequency_data_raw.index = frequency_data_raw.index.tz_localize('UTC')  
    frequency_data_raw.index = frequency_data_raw.index.tz_convert('Europe/Madrid')  
    frequency_data = frequency_data_raw.copy()  
  
    # Load generation data  
    # from 2019-09-29 till 2023-07-01
    with open('generation_data.pkl', 'rb') as file:  
        generation_data = pickle.load(file)  

    return frequency_data, generation_data  
