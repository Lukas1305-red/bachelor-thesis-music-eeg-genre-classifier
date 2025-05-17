function createEEGPlots(filename)
    channel_count = 18; % change if wrong
    
    for i = 1:channel_count
        fprintf('Drawing Channel %d...\n', i)
        plotXDF(filename, i);
    end
    fprintf('EEG Visualizing done!')
end