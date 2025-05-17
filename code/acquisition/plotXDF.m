function plotXDF(filename, channel_number)
    % load the xdf file
    streams = load_xdf(filename)

    for i = 1:length(streams)
        fprintf('Stream %d: %s\n', i, streams{i}.info.name);
    end

    % store eeg streams and time markers
    
    eeg_data = streams{3}.time_series;
    eeg_timestamps = streams{3}.time_stamps;

    song_markers = streams{1}.time_series;
    song_timestamps = streams{1}.time_stamps;

    recording_markers = streams{2}.time_series;
    recording_timestamps = streams{2}.time_stamps;

    % Plot the channel number
    figure;
    plot(eeg_timestamps, eeg_data(channel_number, :))
    xlabel('Time (s)')
    ylabel('EEG Amplitude')
    title(sprintf('EEG Data for Channel %d', channel_number))
    grid on;
    hold on;

    for i = 1:length(song_timestamps)
        xline(song_timestamps(i), 'r', song_markers{2*i}, 'LabelVerticalAlignment','bottom', 'LabelHorizontalAlignment','right');
    end

    for i = 1:length(recording_timestamps)
        xline(recording_timestamps(i), 'c-', recording_markers{i}, 'LabelVerticalAlignment','top', 'LabelHorizontalAlignment','left');
    end

    legend('EEG', 'Song Markers', 'Recording Markers')
    hold off;
       

