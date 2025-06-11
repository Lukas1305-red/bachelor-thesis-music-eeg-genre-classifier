import pyxdf
import numpy as np
import pandas as pd
import os
from scipy.io import savemat

# Constants
PATH_TO_DATA = "../../data/eeg-data"
SONG_COUNT = 50

# Helper functions
def get_channel_names(biosignal_stream):
    ret = []
    acc = map(lambda x: x["label"][0], biosignal_stream["info"]["desc"][0]["channels"][0]["channel"])
    for channel_name in acc:
        ret.append(channel_name)
    return ret

def get_start_from_segment(start, timestamps, prev_pointer):
    start_pointer = prev_pointer
    while True:
      if timestamps[start_pointer] >= start: # found the start timestamp or rather got the closest one
        break
      start_pointer += 1
    print(f"The Start pointer is {start_pointer}")
    return start_pointer

def segment_signal_data(timestamp_list, signal_list, channel_count, music_timestamps):
  # Read out EEG channel data
  data_acc = [] # stores data over the course of all channels and each song i guess idk
  pointer = 0 # pointer that iterates over all the timestamps/data
  for i in range(0, SONG_COUNT*2, 2):
    # get the start and end timestamp of the song
    print(f"======START {i}=====")
    start = music_timestamps[i]
    print(f"Start timestamp: {start}")
    end = music_timestamps[i+1]
    print(f"End timestamp: {end}")

    # find the index in which the timestamp can be found
    pointer = get_start_from_segment(start, timestamp_list, pointer)
    print(f"The pointer is {pointer}")
    pointer_buf = pointer # store the pointer since it is increased steadily

    print("START READING OUT SIGNAL DATA")
    song_data = [] # the eeg data from this specific song (song i)
    for j in range(channel_count):
      print(f"Channel {j} is being read out")
      channel_data = [] # signal data from the channel that is being read out rn
      current_channel = signal_list[:, j] # get the current channel
      while timestamp_list[pointer] < end:
        # append data to the channel data list
        channel_data.append(current_channel[pointer])
        pointer += 1
      pointer = pointer_buf # reset the pointer
      song_data.append(channel_data) # append the channel data to the current song's signal data
    data_acc.append(song_data)
    print(f"======END {i}=====")
  return data_acc

def find_stream(stream_name, data):
    for stream in data:
      if stream["info"]["name"] == [stream_name]:
        print(f"Found the stream {stream_name}!")
        return stream
    print("Couldn't find the stream.")

def store_in_table(columns, segmented_signal_data, table):
  # initialize the rest of the columns in the table
  for channel in columns:
    table[channel] = []

  # now we have to construct the actual values of the columns
  for song_counter in range(SONG_COUNT):
    current_song = segmented_signal_data[song_counter]
    counter = 0
    for channel_data in current_song:
      column_name = columns[counter]
      counter += 1
      table[column_name].append(channel_data)

# Main logic

def create_segment(session_id):
    table = {}
    filename = f"sub-P001_ses-{session_id}_task-Default_run-001_eeg.xdf"
    data, _ = pyxdf.load_xdf(f"{PATH_TO_DATA}/{filename}")

    eeg_stream = find_stream("Headset_EEG_DR250", data=data)
    music_markers_stream = find_stream("MusicMarkers", data=data)
    song_stream = find_stream("Song", data=data)
    temperature_stream = find_stream("Headset_TMP_DR1", data=data)
    ppg_algo_stream = find_stream("Headset_PPG_ALGO_DR25", data=data)
    imu_stream = find_stream("Headset_IMU_DR30", data=data)

    # Read out channel names of each stream
    print("EEG Channels:")
    eeg_channels = get_channel_names(eeg_stream)
    for elem in eeg_channels:
        print(elem)
    print("===========")
    print("IMU Channels:")
    imu_channels = get_channel_names(imu_stream)
    for elem in imu_channels:
        print(elem)
    print("===========")
    print("Temperature Channels:")
    temperature_channels = get_channel_names(temperature_stream)
    for elem in temperature_channels:
        print(elem)
    print("===========")
    print("PPG Channels:")
    ppg_channels = get_channel_names(ppg_algo_stream)
    for elem in ppg_channels:
        print(elem)
    
    # Get timestamps and data
    # Annotation: data holds the real values; timestamps the corresponding time the specific value was measured at
    eeg_data = eeg_stream["time_series"]
    eeg_timestamps = eeg_stream["time_stamps"]

    temperature_data = temperature_stream["time_series"]
    temperature_timestamps = temperature_stream["time_stamps"]

    ppg_algo_data = ppg_algo_stream["time_series"]
    ppg_algo_timestamps = ppg_algo_stream["time_stamps"]

    imu_data = imu_stream["time_series"]
    imu_timestamps = imu_stream["time_stamps"]


    music_markers_data = music_markers_stream["time_series"]
    music_markers_timestamps = music_markers_stream["time_stamps"]

    song_data = song_stream["time_series"]

    table["genre"] = []
    for i in range(len(song_data)-1):
      table["genre"].append(song_data[i][1])
    
    eeg_data_processed = eeg_data[:, :8]
    channel_data_acc = segment_signal_data(timestamp_list=eeg_timestamps, signal_list=eeg_data_processed, channel_count=8, music_timestamps=music_markers_timestamps)
    eeg_channels_col = [f"EEG_{eeg_channels[i]}" for i in range(8)]
    store_in_table(columns=eeg_channels_col, segmented_signal_data=channel_data_acc)

    temperature_data_buffer = segment_signal_data(timestamp_list=temperature_timestamps, signal_list=temperature_data, channel_count=3, music_timestamps=music_markers_timestamps) 
    temp_columns = [f"TEMP_{temperature_channels[i]}" for i in range(3)]
    store_in_table(columns=temp_columns, segmented_signal_data=temperature_data_buffer)

    imu_data_buffer = segment_signal_data(timestamp_list=imu_timestamps, signal_list=imu_data, channel_count=6, music_timestamps=music_markers_timestamps)
    imu_columns = [f"IMU_Ch{imu_channels[i]}" for i in range(6)]
    store_in_table(imu_columns, imu_data_buffer)

    ppg_data_buffer = segment_signal_data(timestamp_list=ppg_algo_timestamps, signal_list=ppg_algo_data, channel_count=23, music_timestamps=music_markers_timestamps)
    ppg_columns = [f"PPG_{ppg_channels[i]}" for i in range(23)]
    print(ppg_columns)
    store_in_table(ppg_columns, ppg_data_buffer)

    return table

def main():
    rows = []
    for i in range(SONG_COUNT):
      rows.append(f"Song{i}")
    for i in range(1):
      session_number = str(i).zfill(4)
      table = create_segment(session_number)
      df = pd.DataFrame(table)
      df.index = rows

      # Save the table
      os.makedirs(f"{PATH_TO_DATA}/signal_data", exist_ok=True)

      # create a folder for numpy data
      os.makedirs(f"{PATH_TO_DATA}/signal_data/numpy", exist_ok=True)
      np.save(f"{PATH_TO_DATA}/signal_data/numpy/{session_number}.npy", df.to_numpy())
      # create a folder for matlab data
      os.makedirs(f"{PATH_TO_DATA}/signal_data/matlab", exist_ok=True)
      savemat(f"{PATH_TO_DATA}/signal_data/matlab/{session_number}.mat", table)

      os.makedirs(f"{PATH_TO_DATA}/signal_data/csv", exist_ok=True)
      df.to_csv(f"{PATH_TO_DATA}/signal_data/csv/{session_number}.csv")
    


if __name__ == "__main__":
    main()