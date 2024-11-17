clc; clear; close all;

% Load the data from the CSV file
filename = 'csv/emotions_tracking.csv';
data = readtable(filename);

% Extract the unique emotions and persons
unique_emotions = unique(data.Emotion);
unique_persons = unique(data.Person_ID);

emotion_counts = zeros(length(unique_emotions), length(unique_persons));

% Count the number of occurrences of each emotion for each person
for i = 1:length(unique_emotions)
    for j = 1:length(unique_persons)
        emotion_counts(i, j) = sum(strcmp(data.Emotion, unique_emotions{i}) & data.Person_ID == unique_persons(j));
    end
end

% Find the maximum count for each emotion
max_counts_per_emotion = max(emotion_counts, [], 2);

% Plot the bar chart
fig = figure;
b = bar(emotion_counts, 'stacked');
set(gca, 'XTickLabel', unique_emotions);
xlabel('Emotion');
ylabel('Count');
title('Emotion Comparison Between Participants');
legend(arrayfun(@(x) sprintf('Person %d', x), unique_persons, 'UniformOutput', false));

grid on;

colors = [135/255, 206/255, 250/255; 255/255, 127/255, 80/255];
for k = 1:length(b)
    b(k).FaceColor = colors(k, :);
end


% Add the maximum count as text on top of each bar group
for i = 1:length(unique_emotions)
    text(i, sum(emotion_counts(i, :)), sprintf('%d', max_counts_per_emotion(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
end

saveas(fig, "figures/emotions_tracking.png");