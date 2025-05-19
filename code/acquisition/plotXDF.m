function plotXDF(channel_number, eeg_stream, song_stream, music_markers_stream, draw_segments)
    
    eeg_data = eeg_stream.time_series;
    eeg_timestamps = eeg_stream.time_stamps;

    song_markers = song_stream.time_series;
    song_timestamps = song_stream.time_stamps;

    recording_markers = music_markers_stream.time_series;
    recording_timestamps = music_markers_stream.time_stamps;

    % Plot the channel number
    figure;
    plot(eeg_timestamps, eeg_data(channel_number, :))
    xlabel('Time (s)')
    ylabel('EEG Amplitude')
    title(sprintf('EEG Data for Channel %d', channel_number))
    grid on;
    hold on;
    
    if draw_segments == true 
        for i = 1:length(song_timestamps)
            xline(song_timestamps(i), 'r', song_markers{2*i}, 'LabelVerticalAlignment','bottom', 'LabelHorizontalAlignment','right');
        end

        for i = 1:length(recording_timestamps)
            xline(recording_timestamps(i), 'c-', recording_markers{i}, 'LabelVerticalAlignment','top', 'LabelHorizontalAlignment','left');
        end
        legend('EEG', 'Song Markers', 'Recording Markers')
    end
    hold off;
       

