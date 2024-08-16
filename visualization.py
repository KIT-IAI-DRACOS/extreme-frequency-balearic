import matplotlib.pyplot as plt  
import pandas as pd  
import numpy as np  
  
def visualize_results(classifier, X_test, y_test, filtered_data):  
    # Plot feature importance  
    plot_feature_importance(classifier)  
      
    # Plot prediction results  
    plot_predictions(classifier, X_test, y_test, filtered_data)  
      
    # Plot temporal distance to next event for false positives  
    plot_temporal_distance_to_next_event(classifier, X_test, y_test, filtered_data)  
  
def plot_feature_importance(classifier):  
    fi = pd.DataFrame(data=classifier.feature_importances_, index=classifier.feature_names_in_, columns=['importance'])  
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance')  
    plt.figure(figsize=(30, 30))  
    plt.show()  
  
def plot_predictions(classifier, X_test, y_test, filtered_data):  
    plot_prediction = pd.DataFrame(X_test.copy())  
    plot_prediction["prediction"] = classifier.predict(X_test)  
    df = filtered_data.merge(plot_prediction[['prediction']], how='left', left_index=True, right_index=True)  
  
    ax = df.loc[(df.index > '2023-06-25') & (df.index <= '2023-06-27')]['event_pred'].plot(figsize=(15, 5), title='2023-06-29')  
    df.loc[(df.index > '2023-06-25') & (df.index <= '2023-06-27')]['prediction'].plot(style='.')  
    plt.legend(['Truth Data', 'Prediction'])  
    plt.show()  
  
def plot_temporal_distance_to_next_event(classifier, X_test, y_test, filtered_data):  
    histogram_df = pd.DataFrame(X_test.copy())  
    histogram_df["day"] = histogram_df["dayofmonth"]  
    histogram_df['datetime'] = pd.to_datetime(histogram_df[['year', 'month', 'day', 'hour', 'minute']])  
    histogram_df["event_pred"] = y_test  
    false_positives_indices = np.where((classifier.predict(X_test) != y_test) & (classifier.predict(X_test) != 1))[0]  
    event_datetimes = histogram_df[histogram_df['event_pred'] != 1]['datetime']  
    distances_to_next_event = []  
  
    for index in false_positives_indices:  
        fp_datetime = histogram_df.iloc[index]['datetime']  
        time_differences = (event_datetimes - fp_datetime).dt.total_seconds() / 60.0  
        future_events_within_30 = time_differences[(time_differences > 0) & (time_differences <= 180)]  
        if not future_events_within_30.empty:  
            min_diff_to_next_event = future_events_within_30.min()  
            distances_to_next_event.append(min_diff_to_next_event)  
      
    min_distance = min(distances_to_next_event)
    max_distance = max(distances_to_next_event)  
    range_of_distances = max_distance - min_distance  
    num_bins = int(np.ceil(range_of_distances / 5))  # 5 minutes per bin  
  
    plt.hist(distances_to_next_event, bins=num_bins, edgecolor='k')  
    plt.xlabel('Minutes to Next Event')  
    plt.ylabel('Count of False Positives')  
    plt.title('Temporal Distance to Next Event for False Positives')  
    xticks_range = np.arange(0, max_distance + 15, 15)  # Adjust the step to 15 minutes  
    plt.xticks(xticks_range)  
    plt.savefig('plots/temporal_distance_false_positives.pdf', format='pdf')  
    plt.show()  
  
def plot_temporal_distance_to_random_false_events(classifier, X_test, y_test, filtered_data):  
    histogram_df = pd.DataFrame(X_test.copy())  
    histogram_df["day"] = histogram_df["dayofmonth"]  
    histogram_df['datetime'] = pd.to_datetime(histogram_df[['year', 'month', 'day', 'hour', 'minute']])  
    histogram_df["event_pred"] = y_test  
    num_instances = len(y_test)  
    random_false_positives_indices = np.random.choice(num_instances, 40000, replace=False)  
    event_datetimes = histogram_df[histogram_df['event_pred'] != 1]['datetime']  
    distances_to_next_event = []  
  
    for index in random_false_positives_indices:  
        fp_datetime = histogram_df.iloc[index]['datetime']  
        time_differences = (event_datetimes - fp_datetime).dt.total_seconds() / 60.0  
        future_events_within_30 = time_differences[(time_differences > 0) & (time_differences <= 180)]  
        if not future_events_within_30.empty:  
            min_diff_to_next_event = future_events_within_30.min()  
            distances_to_next_event.append(min_diff_to_next_event)  
      
    distances_to_next_event = [d for d in distances_to_next_event if d is not None]  
    min_distance = min(distances_to_next_event)  
    max_distance = max(distances_to_next_event)  
    num_bins = int(np.ceil(180 / 5))  # 5 minutes per bin  
  
    plt.hist(distances_to_next_event, bins=num_bins, edgecolor='k')  
    plt.xlabel('Minutes to Next Event')  
    plt.ylabel('Count of Random False Positives')  
    plt.title('Temporal Distance to Next Event for Randomly Generated False Positives')  
    xticks_range = np.arange(0, max_distance + 15, 15)  # Adjust the step to 15 minutes  
    plt.xticks(xticks_range)  
    plt.savefig('plots/temporal_distance_random_false_positives.pdf', format='pdf')  
    plt.show()  
  
def plot_boxplots_for_false_positives(classifier, X_test, y_test, frequency_data):  
    histogram_df = pd.DataFrame(X_test.copy())  
    histogram_df["day"] = histogram_df["dayofmonth"]  
    histogram_df['datetime'] = pd.to_datetime(histogram_df[['year', 'month', 'day', 'hour', 'minute']])  
    histogram_df["event_pred"] = y_test  
    false_positives_indices_negative_events = np.where((classifier.predict(X_test) != y_test) & (classifier.predict(X_test) == 0))[0]  
    false_positives_indices_positive_events = np.where((classifier.predict(X_test) != y_test) & (classifier.predict(X_test) == 2))[0]  
      
    positive_events_fp = histogram_df.iloc[false_positives_indices_positive_events]  
    negative_events_fp = histogram_df.iloc[false_positives_indices_negative_events]
  
    plt.figure(figsize=(12, 6))  
    for i, (fp_events, title) in enumerate(zip([negative_events_fp, positive_events_fp], ['Negative Events', 'Positive Events'])):  
        ax = plt.subplot(1, 2, i+1)  
        fp_events['event_pred'] = np.where(fp_events['event_pred'] == 0, 'Negative', 'Positive')  
        frequency_data_filtered = frequency_data[frequency_data.index.isin(fp_events.index)]  
        merged_df = fp_events.merge(frequency_data_filtered[['demand_real']], how='left', left_index=True, right_index=True)  
        merged_df.boxplot(column='demand_real', by='event_pred', ax=ax)  
        plt.title(f'False Positives for {title}')  
        plt.suptitle('')  
        plt.xlabel('Event Type')  
        plt.ylabel('Real Demand')  
      
    plt.tight_layout()  
    plt.savefig('plots/boxplots_false_positives.pdf', format='pdf')  
    plt.show()  
  
def plot_daily_event_distribution(filtered_data):  
    daily_event_counts = filtered_data.resample('D')['event_pred'].sum()  
    plt.figure(figsize=(12, 6))  
    daily_event_counts.plot(kind='bar')  
    plt.xlabel('Date')  
    plt.ylabel('Event Count')  
    plt.title('Daily Event Distribution')  
    plt.savefig('plots/daily_event_distribution.pdf', format='pdf')  
    plt.show()  
  
def plot_hourly_event_distribution(filtered_data):  
    filtered_data['hour'] = filtered_data.index.hour  
    hourly_event_counts = filtered_data.groupby('hour')['event_pred'].sum()  
    plt.figure(figsize=(12, 6))  
    hourly_event_counts.plot(kind='bar')  
    plt.xlabel('Hour of the Day')  
    plt.ylabel('Event Count')  
    plt.title('Hourly Event Distribution')  
    plt.savefig('plots/hourly_event_distribution.pdf', format='pdf')  
    plt.show()  
  
def plot_weekly_event_distribution(filtered_data):  
    filtered_data['week'] = filtered_data.index.isocalendar().week  
    weekly_event_counts = filtered_data.groupby('week')['event_pred'].sum()  
    plt.figure(figsize=(12, 6))  
    weekly_event_counts.plot(kind='bar')  
    plt.xlabel('Week of the Year')  
    plt.ylabel('Event Count')  
    plt.title('Weekly Event Distribution')  
    plt.savefig('plots/weekly_event_distribution.pdf', format='pdf')  
    plt.show()  
  
def plot_event_duration(filtered_data):  
    filtered_data['event_duration'] = filtered_data['event_pred'].diff().fillna(0).ne(0).cumsum()  
    event_durations = filtered_data.groupby('event_duration')['event_pred'].sum().value_counts()  
    plt.figure(figsize=(12, 6))  
    event_durations.plot(kind='bar')  
    plt.xlabel('Event Duration (minutes)')  
    plt.ylabel('Frequency')  
    plt.title('Event Duration Distribution')  
    plt.savefig('plots/event_duration_distribution.pdf', format='pdf')  
    plt.show()  



