import pyxdf
import numpy as np
import pandas as pd
import os
from scipy.io import savemat

from code.denoising.denoising import (
    bandpass,
    notch_filter,
    artifact_removal_ica,
    apply_z_normalization,
)

from code.feature_extraction.feature_extraction import feature_extraction

# MARK: Constants
PATH_TO_DATA = "data/eeg-data"
PATH_TO_DATA_ROOT = "data"


# MARK: Helper functions
def get_channel_names(biosignal_stream):
    ret = []
    acc = map(
        lambda x: x["label"][0],
        biosignal_stream["info"]["desc"][0]["channels"][0]["channel"],
    )
    for channel_name in acc:
        ret.append(channel_name)
    return ret


def get_start_from_segment(start, timestamps, prev_pointer):
    start_pointer = prev_pointer
    while True:
        if timestamps[start_pointer] >= start:
            break
        start_pointer += 1
    return start_pointer


def segment_signal_data(
    timestamp_list, signal_list, channel_count, music_timestamps, song_count
):
    # Read out EEG channel data
    data_acc = []  # stores data over the course of all channels and each song i guess idk
    pointer = 0  # pointer that iterates over all the timestamps/data
    for i in range(0, song_count * 2, 2):
        # get the start and end timestamp of the song
        start = music_timestamps[i]
        end = music_timestamps[i + 1]

        # find the index in which the timestamp can be found
        pointer = get_start_from_segment(start, timestamp_list, pointer)
        pointer_buf = pointer  # store the pointer since it is increased steadily

        song_data = []  # the eeg data from this specific song (song i)
        for j in range(channel_count):
            channel_data = []  # signal data from the channel that is being read out rn
            current_channel = signal_list[:, j]  # get the current channel
            while timestamp_list[pointer] < end:
                # append data to the channel data list
                channel_data.append(current_channel[pointer])
                pointer += 1
            pointer = pointer_buf  # reset the pointer
            song_data.append(
                channel_data
            )  # append the channel data to the current song's signal data
        data_acc.append(song_data)
    return data_acc


def find_stream(stream_name, data):
    for stream in data:
        if stream["info"]["name"] == [stream_name]:
            print(f"Found the stream {stream_name}!")
            return stream
    print("Couldn't find the stream.")


def debug_print_shapes(segmented_signal_data, song_count):
    for i in range(song_count):
        print(
            f"This is the shape of song {i + 1}: {np.shape(segmented_signal_data[i])}"
        )


def store_in_table(columns, segmented_signal_data, table, song_count):
    # initialize the rest of the columns in the table
    for channel in columns:
        if channel != "EEG_Ch3":
            table[channel] = []

    # now we have to construct the actual values of the columns
    for song_counter in range(song_count):
        current_song = segmented_signal_data[song_counter]
        counter = 0
        for channel_data in current_song:
            column_name = columns[counter]
            counter += 1
            if column_name != "EEG_Ch3":
                table[column_name].append(channel_data)


def filter_eeg(eeg_data, sample_rate):
    filtered = []
    for i in range(7):
        channel = eeg_data[:, i]
        filtered_channel = bandpass(channel, [1, 30], sample_rate)
        filtered_channel = notch_filter(filtered_channel, 50, sample_rate)
        filtered.append(filtered_channel)
    return np.transpose(filtered)


def remove_artefacts(eeg_data):
    cleaned_data = artifact_removal_ica(np.transpose(eeg_data))
    return np.transpose(cleaned_data)


def extract_features_from_table(eeg_df: pd.DataFrame, column_names):
    feature_rows = []
    for _, row in eeg_df.iterrows():
        buf = []
        for column in column_names:
            buf.append(row[column])
        features_song = feature_extraction(song_data=buf)
        feature_rows.append(features_song)
    feature_df = pd.DataFrame(feature_rows)
    return feature_df


# MARK: Main logic


def create_segment(session_id, song_count_param, only_eeg_segments, filter_on):
    table = {}
    sessions_without_ppg = ["000", "001", "002", "003", "004"]
    filename = f"sub-P001_ses-S{session_id}_task-Default_run-001_eeg.xdf"
    data, _ = pyxdf.load_xdf(f"{PATH_TO_DATA}/{filename}")

    eeg_stream = find_stream("Headset_EEG_DR250", data=data)
    music_markers_stream = find_stream("MusicMarkers", data=data)
    song_stream = find_stream("Song", data=data)
    temperature_stream = find_stream("Headset_TMP_DR1", data=data)
    ppg_algo_stream = find_stream("Headset_PPG_ALGO_DR25", data=data)
    imu_stream = find_stream("Headset_IMU_DR30", data=data)

    # Read out channel names of each stream
    eeg_channels = get_channel_names(eeg_stream)
    imu_channels = get_channel_names(imu_stream)
    temperature_channels = get_channel_names(temperature_stream)
    if session_id not in sessions_without_ppg:
        ppg_channels = get_channel_names(ppg_algo_stream)

    # Get timestamps and data
    # Annotation: data holds the real values; timestamps the corresponding time the specific value was measured at
    eeg_data = eeg_stream["time_series"]
    eeg_timestamps = eeg_stream["time_stamps"]

    temperature_data = temperature_stream["time_series"]
    temperature_timestamps = temperature_stream["time_stamps"]

    if session_id not in sessions_without_ppg:
        ppg_algo_data = ppg_algo_stream["time_series"]
        ppg_algo_timestamps = ppg_algo_stream["time_stamps"]

    imu_data = imu_stream["time_series"]
    imu_timestamps = imu_stream["time_stamps"]

    # _ = music_markers_stream["time_series"]
    music_markers_timestamps = music_markers_stream["time_stamps"]

    song_data = song_stream["time_series"]

    table["genre"] = []
    # song_count = song_data
    for i in range(song_count_param):
        table["genre"].append(song_data[i][1])

    eeg_data_processed = np.delete(eeg_data[:, :8], 2, axis=1)
    if filter_on:
        eeg_data_denoised = filter_eeg(eeg_data_processed, sample_rate=250)
        eeg_data_denoised = remove_artefacts(eeg_data_denoised)
    channel_data_acc = segment_signal_data(
        timestamp_list=eeg_timestamps,
        signal_list=eeg_data_denoised if filter_on else eeg_data_processed,
        channel_count=7,
        music_timestamps=music_markers_timestamps,
        song_count=song_count_param,
    )
    if False:
        buf = []
        for song in channel_data_acc:
            acc = apply_z_normalization(song_data=song)
            buf.append(acc)
        channel_data_acc = buf
    eeg_channels_col = [f"EEG_{eeg_channels[i]}" for i in range(8)]
    eeg_channels_col.remove("EEG_Ch3")
    store_in_table(
        columns=eeg_channels_col,
        segmented_signal_data=channel_data_acc,
        table=table,
        song_count=song_count_param,
    )
    if not only_eeg_segments:
        temperature_data_buffer = segment_signal_data(
            timestamp_list=temperature_timestamps,
            signal_list=temperature_data,
            channel_count=3,
            music_timestamps=music_markers_timestamps,
            song_count=song_count_param,
        )
        temp_columns = [f"TEMP_{temperature_channels[i]}" for i in range(3)]
        store_in_table(
            columns=temp_columns,
            segmented_signal_data=temperature_data_buffer,
            table=table,
            song_count=song_count_param,
        )

        imu_data_buffer = segment_signal_data(
            timestamp_list=imu_timestamps,
            signal_list=imu_data,
            channel_count=6,
            music_timestamps=music_markers_timestamps,
            song_count=song_count_param,
        )
        imu_columns = [f"IMU_Ch{imu_channels[i]}" for i in range(6)]
        store_in_table(
            imu_columns, imu_data_buffer, table=table, song_count=song_count_param
        )

        if session_id not in sessions_without_ppg:
            ppg_data_buffer = segment_signal_data(
                timestamp_list=ppg_algo_timestamps,
                signal_list=ppg_algo_data,
                channel_count=23,
                music_timestamps=music_markers_timestamps,
                song_count=song_count_param,
            )
            ppg_columns = [f"PPG_{ppg_channels[i]}" for i in range(23)]
            store_in_table(
                ppg_columns, ppg_data_buffer, table=table, song_count=song_count_param
            )

    return table


# MARK: Main method


def main(only_eeg_segments, filter_on, feature_extraction_on):
    SONG_COUNT = 50
    DIR_NAME = "signal_data"
    if only_eeg_segments:
        DIR_NAME = "eeg_data_v7_denoised"
    if only_eeg_segments and feature_extraction_on:
        DIR_NAME = "eeg_data_features"
    os.makedirs(f"{PATH_TO_DATA_ROOT}/{DIR_NAME}", exist_ok=True)
    for i in range(20):
        print(f"SEGMENTING SESSION {i}...")
        count = SONG_COUNT
        if i == 10:
            count -= 1
        rows = []
        for j in range(count):
            rows.append(f"Song{j}")
        session_number = str(i).zfill(3)
        table = create_segment(
            session_number,
            song_count_param=count,
            only_eeg_segments=only_eeg_segments,
            filter_on=filter_on,
        )
        df = pd.DataFrame(table)
        df.index = rows

        if feature_extraction_on:
            column_names = [f"EEG_Ch{i + 1}" for i in range(8)]
            column_names.remove("EEG_Ch3")
            print(f"START FEATURE EXTRACTION FOR {session_number}...")
            df_features = extract_features_from_table(
                eeg_df=df, column_names=column_names
            )
            df_features.index = rows
            os.makedirs(f"{PATH_TO_DATA_ROOT}/{DIR_NAME}/features", exist_ok=True)
            df_features.to_pickle(
                f"{PATH_TO_DATA_ROOT}/{DIR_NAME}/features/{session_number}.pkl"
            )
            print(f"SUCCESSFULLY EXTRACTED FEATURES FOR {session_number}...")

        # create a folder for numpy data
        os.makedirs(f"{PATH_TO_DATA_ROOT}/{DIR_NAME}/numpy", exist_ok=True)
        np.save(
            f"{PATH_TO_DATA_ROOT}/{DIR_NAME}/numpy/{session_number}.npy", df.to_numpy()
        )
        # create a folder for matlab data
        os.makedirs(f"{PATH_TO_DATA_ROOT}/{DIR_NAME}/matlab", exist_ok=True)
        savemat(f"{PATH_TO_DATA_ROOT}/{DIR_NAME}/matlab/{session_number}.mat", table)
        # create a folder for pickle data
        os.makedirs(f"{PATH_TO_DATA_ROOT}/{DIR_NAME}/pickle", exist_ok=True)
        df.to_pickle(f"{PATH_TO_DATA_ROOT}/{DIR_NAME}/pickle/{session_number}.pkl")

        print(f"SUCCESSFULLY SAVED SESSION {session_number}")


if __name__ == "__main__":
    main(only_eeg_segments=True, filter_on=True, feature_extraction_on=True)
