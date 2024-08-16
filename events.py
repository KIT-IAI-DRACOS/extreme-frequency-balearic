import pandas as pd
import numpy as np

def get_events(frequency_data, threshold, max_gap_length=5, event_start=5, event_end=5, allow_overlap=False):
    is_event = threshold
    event_times = frequency_data.index[is_event]
    event_groups = []
    current_group = [event_times[0]]

    for time in event_times[1:]:
        if (time - current_group[-1]) <= pd.Timedelta(minutes=max_gap_length):
            current_group.append(time)
        else:
            event_groups.append(current_group)
            current_group = [time]

    if current_group:
        event_groups.append(current_group)

    # Expand event duration 
    events_data = []

    for group in event_groups:
        start = group[0] - pd.Timedelta(minutes=event_start)
        end = group[-1] + pd.Timedelta(minutes=event_end)
        # Ensure no overlap with next event
        if not allow_overlap:
            if events_data and start < events_data[-1]['End']:
                start = events_data[-1]['End']
        events_data.append({'Start': start, 'End': end})

    events = pd.DataFrame(events_data)

    # Ensure events don't exceed the time frame bounds
    #
    # ['Start'] = np.maximum(events['Start'], frequency_data.index[0])
    events['End'] = np.minimum(events['End'], frequency_data.index[-1])

    return events


def get_events_for_forecasting(frequency_data, is_event, max_gap_length=5):
    # Identify when the frequency is above the threshold
    #is_event = frequency_data > threshold
    times = frequency_data.index

    event_groups = []
    current_group = []
    last_time_above_threshold = None

    for time, above_threshold in zip(times, is_event):
        if above_threshold:
            if not current_group or (time - last_time_above_threshold) <= pd.Timedelta(minutes=max_gap_length):
                current_group.append(time)
            else:
                event_groups.append(current_group)
                current_group = [time]
            last_time_above_threshold = time
        elif current_group and (time - last_time_above_threshold) > pd.Timedelta(minutes=max_gap_length):
            event_groups.append(current_group)
            current_group = []

    if current_group:
        event_groups.append(current_group)

    # Create event data
    events_data = [{'Start': group[0], 'End': group[-1]} for group in event_groups]
    events = pd.DataFrame(events_data)
    return events
