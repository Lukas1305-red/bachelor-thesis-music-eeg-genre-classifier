function createEEGPlots(filename, draw_segments)
    channel_count = 18; % change if wrong
    streams = load_xdf(filename);
    
    % Get information about the streams
    for i = 1:length(streams)
        fprintf('Stream %d: %s\n', i, streams{i}.info.name);
        switch streams{i}.info.name
            case Headset_EEG_DR250
                eeg_stream = streams{i};
            case MusicMarkers
                music_markers_stream = streams{i};
            case Song
                song_stream = streams{i};
        end
    end

    for i = 1:channel_count
        fprintf('Drawing Channel %d...\n', i)
        plotXDF(i, eeg_stream, song_stream, music_markers_stream, draw_segments);
    end
    fprintf('EEG Visualizing done!\n')
end