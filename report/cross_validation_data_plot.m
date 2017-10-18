data = textread('cross_validation_data.csv', '%s', 'whitespace', ',');
data = reshape(data(3:end), 2, (numel(data) - 2) / 2)';

names = data(:, 1);
results = 100 .* cellfun(@str2num, data(:, 2));

figure;
bar(results);
set(gca, 'XTickLabel', names);
xtickangle(45);
ylabel('Performance (%)');
grid on;

