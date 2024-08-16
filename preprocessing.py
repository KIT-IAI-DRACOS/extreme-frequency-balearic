import pandas as pd  
import numpy as np  
import events
from holidays_es import Province, HolidaySpain  
  
def get_events(frequency_data):  
    positive_events = events.get_events(frequency_data=frequency_data, threshold=(frequency_data > 150), max_gap_length=5, event_end=0, event_start=5)  
    negative_events = events.get_events(frequency_data=frequency_data, threshold=(frequency_data < -150), max_gap_length=5, event_end=0, event_start=5)  
    return positive_events, negative_events  
  
def preprocess_data(frequency_data, generation_data, positive_events, negative_events):  
    
    scheduled_columns = [
    'generation_coal', 
    'generation_diesel_engines',
    'generation_gas_turbine', 
    'generation_combined_cycle',
    'generation_cogeneration',
    'generation_waste',
    'generation_auxiliary']

    dynamic_columns = [
        'generation_solar',
        'generation_wind',
        'generation_other_renewables',
    ]

    columns_to_drop = [
        'generation_waste',
        'generation_cogeneration',
        'generation_auxiliary',
        'generation_wind',
        'generation_other_renewables',
    ]

    generation_data['generation_dynamic'] = generation_data[dynamic_columns].sum(axis=1)
    generation_data['generation_scheduled'] = generation_data[scheduled_columns].sum(axis=1)

    generation_data = generation_data.drop(columns=columns_to_drop)
    #generation_data = generation_data.drop(columns=dynamic_columns)
    frequency_data = frequency_data.to_frame()
    frequency_data.rename(columns={0: 'frequency'}, inplace=True)
    frequency_data_profile = frequency_data.copy()

    frequency_data_profile['time'] = frequency_data_profile.index.time

    # group by the 'time' column and calculate the mean frequency for each time
    data_day_profile = frequency_data_profile.groupby('time').mean()

    data_day_profile = data_day_profile.rename(columns={"frequency": 'expected_frequency'})

    # map the expected_frequency values back to the original DataFrame
    frequency_data['expected_frequency'] = frequency_data_profile['time'].map(data_day_profile['expected_frequency'])
    frequency_data['ema_5_minutes'] = frequency_data['frequency'].ewm(span=300, adjust=True).mean()
    frequency_data['ema_6_seconds'] = frequency_data['frequency'].ewm(span=6, adjust=True).mean()
    frequency_data['ema_1_minute'] = frequency_data['frequency'].ewm(span=60, adjust=True).mean()
    alpha = 0.1
    frequency_data['squared_deviations_5_minutes'] = (frequency_data['frequency'] - frequency_data['ema_5_minutes'])**2
    frequency_data['emv_5_minutes'] = frequency_data['squared_deviations_5_minutes'].ewm(alpha=alpha, adjust=False).mean()
    frequency_data['squared_deviations_6_seconds'] = (frequency_data['frequency'] - frequency_data['ema_6_seconds'])**2
    frequency_data['emv_6_seconds'] = frequency_data['squared_deviations_6_seconds'].ewm(alpha=alpha, adjust=False).mean()
    frequency_data['squared_deviations_1_minute'] = (frequency_data['frequency'] - frequency_data['ema_1_minute'])**2
    frequency_data['emv_1_minute'] = frequency_data['squared_deviations_1_minute'].ewm(alpha=alpha, adjust=False).mean()
    frequency_data["ramp_frequency_6_seconds"] = frequency_data["frequency"].diff(periods = 6)
    frequency_data["ramp_frequency_30_seconds"] = frequency_data["frequency"].diff(periods = 30)
    generation_data["During_Event"] = 0
    start = frequency_data.index.min()
    end = frequency_data.index.max()

    five_minute_intervals = pd.date_range(start=start, end=end, freq='5T')

    # put the frequency data in a 5 minute interval
    frequency_data_5min = frequency_data[frequency_data.index.isin(five_minute_intervals)]
    # reset the index
    generation_data_reset = generation_data.reset_index()
    frequency_data_5min_df = frequency_data_5min.reset_index()

    # rename time column
    generation_data_reset.rename(columns={generation_data_reset.columns[0]: 'Time'}, inplace=True)
    frequency_data_5min_df.rename(columns={frequency_data_5min_df.columns[0]: 'Time'}, inplace=True)

    #merge frequency and generation data
    merged_data = pd.merge(generation_data_reset, frequency_data_5min_df, on='Time', how='left')
    merged_data = merged_data.set_index('Time')
    merged_data.index = pd.to_datetime(merged_data.index)
    # identify the first and last dates
    first_date = merged_data.index.min().date()
    last_date = merged_data.index.max().date()

    # exclude the first and last day
    filtered_data = merged_data[(merged_data.index.date > first_date) & (merged_data.index.date < last_date)]
    # set the during event feature to 1 for the positive and negative events

    for _, event in positive_events.iterrows():
        start, end = event['Start'] + pd.Timedelta(minutes=5), event['End']
    
        filtered_data.loc[start:end, 'During_Event'] = 1

    for _, event in negative_events.iterrows():
        start, end = event['Start'] + pd.Timedelta(minutes=5), event['End']
    
        filtered_data.loc[start:end, 'During_Event'] = 1
    # 0 = negative event, 1 = no event, 2 = positive event

    # event_pred is the feature we need to predict

    filtered_data["event_pred"] = 1

    for _, event in positive_events.iterrows():
        start = event['Start']
        pred = start + pd.Timedelta(minutes=5)
    
        filtered_data.loc[start:pred, 'event_pred'] = 2

    for _, event in negative_events.iterrows():
        start = event['Start']
        pred = start + pd.Timedelta(minutes=5)
    
        filtered_data.loc[start:pred, 'event_pred'] = 0
    # remove unnecessary columns / mostly 0

    for data in ['tnr', 'trn', 'mallorca-ibiza_link', 'ibiza-formentera_link', 'mallorca-menorca_link']:
                if data in filtered_data.columns:
                    filtered_data = filtered_data.drop(data, axis=1)
    filtered_data["demand_programmed_5min"] = filtered_data["demand_programmed"].shift(periods=-1)
    filtered_data["demand_forecast_5min"] = filtered_data["demand_forecast"].shift(periods=-1)
    filtered_data['next_5_minutes_expected_frequency'] = filtered_data["expected_frequency"].shift(periods=-1)
    filtered_data["demand_programmed_10min"] = filtered_data["demand_programmed"].shift(periods=-2)
    filtered_data["demand_forecast_10min"] = filtered_data["demand_forecast"].shift(periods=-2)
    filtered_data["demand_programmed_15min"] = filtered_data["demand_programmed"].shift(periods=-3)
    filtered_data["demand_forecast_15min"] = filtered_data["demand_forecast"].shift(periods=-3)
    # because of the shift the last few rows are NaN

    filtered_data.drop(filtered_data.tail(3).index,inplace=True)
  
    # Split data  
    filtered_data = create_features(filtered_data)  
    filtered_data = create_sawtooth_feature(filtered_data)  
    filtered_data = calculate_ramp_rates(filtered_data)  
    filtered_data = create_holidays(filtered_data)  
  
    # we have added noise to identify unimportant features using shap
    # now we remove some features because these were less important than the noise
    unimportant_features = ["ramp_demand_forecast", "demand_forecast_5min", "ramp_demand_programmed_15min", "ramp_demand_forecast_15min", "ramp_demand_forecast_5min", "demand_programmed_10min", "ramp_demand_forecast_10min", "holiday", "demand_forecast_10min", "ramp_generation_coal", "ramp_demand_programmed", "ramp_demand_programmed_10min", "quarter", "ramp_demand_programmed_15min"]
    filtered_data.drop(columns=unimportant_features ,axis = 1, inplace=True)
      
    return filtered_data  
  
def create_features(df):  
    df = df.copy()  
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek  
    df['quarter'] = df.index.quarter  
    df['month'] = df.index.month  
    df['year'] = df.index.year  
    df['dayofyear'] = df.index.dayofyear  
    df['dayofmonth'] = df.index.day  
    df['weekofyear'] = df.index.isocalendar().week  
    df['minute'] = df.index.minute  
    return df  
  
def create_sawtooth_feature(df):  
    df = df.copy()  
    df["sawtooth_real_programmed"] = df["demand_real"] - df["demand_programmed"]  
    df["sawtooth_real_forecast"] = df["demand_real"] - df["demand_forecast"]  
    return df  
  
def calculate_ramp_rates(df):  
    df = df.copy()  
    for column in df.columns:  
        if column.startswith('generation_') or column.startswith('balearic') or column.startswith('demand'):  
            df['ramp_' + column] = df[column].diff()  
    return df  
  
def create_holidays(df, province=Province.BALEARES):  
    df = df.copy()  
    df["holiday"] = 0  
    start_year = df.index.year.min()  
    end_year = df.index.year.max()  
    for year in range(start_year, end_year + 1):  
        holiday_spain = HolidaySpain(province=province, year=year)  
        for holiday in holiday_spain.holidays:  
            df.loc[df.index.date == holiday.date, "holiday"] = 1  
    return df  
  
def split_data(filtered_data):  
    filtered_data['week_range'] = filtered_data.index.to_period('W')  
    total_weeks = filtered_data['week_range'].nunique()  
    train_weeks_count = int(total_weeks * 0.6)  
    validate_weeks_count = int(total_weeks * 0.2)  
    test_weeks_count = total_weeks - train_weeks_count - validate_weeks_count  
  
    weeks_and_classes = filtered_data.groupby('week_range')['event_pred'].unique()  
    shuffled_weeks = weeks_and_classes.sample(frac=1, random_state=42).index.tolist()  
  
    train_weeks = shuffled_weeks[:train_weeks_count]  
    validate_weeks = shuffled_weeks[train_weeks_count:train_weeks_count+validate_weeks_count]  
    test_weeks = shuffled_weeks[train_weeks_count+validate_weeks_count:]  
  
    train_data = filtered_data[filtered_data['week_range'].isin(train_weeks)]  
    validate_data = filtered_data[filtered_data['week_range'].isin(validate_weeks)]  
    test_data = filtered_data[filtered_data['week_range'].isin(test_weeks)]  
  
    train_data = train_data.sample(frac=1, random_state=2)  
    validate_data = validate_data.sample(frac=1, random_state=2)  
    test_data = test_data.sample(frac=1, random_state=2)  
  
    train_data.drop(columns=["week_range"], axis=1, inplace=True)  
    validate_data.drop(columns=["week_range"], axis=1, inplace=True)  
    test_data.drop(columns=["week_range"], axis=1, inplace=True)  
  
    train_data.dropna(inplace=True)  
    test_data.dropna(inplace=True)  
    validate_data.dropna(inplace=True)  
  
    X_train = train_data.drop(columns=["event_pred"], axis=1)  
    y_train = train_data["event_pred"]  
    X_validate = validate_data.drop(columns=["event_pred"], axis=1)  
    y_validate = validate_data["event_pred"]  
    X_test = test_data.drop(columns=["event_pred"], axis=1)  
    y_test = test_data["event_pred"]  
  
    X_test = X_test.astype(float)  
    y_test = y_test.astype(float)  
    X_train = X_train.astype(float)  
    y_train = y_train.astype(float)  
    X_validate = X_validate.astype(float)  
    y_validate = y_validate.astype(float)  
  
    return X_train, y_train, X_validate, y_validate, X_test, y_test  


