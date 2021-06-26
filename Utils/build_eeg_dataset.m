%% download the EEG alcoholism dataset
if ~exist('../Data/eeg_full/', 'dir')
    fprintf('Downloading the dataset...\n');
    websave('../Data/eeg_full.tar', 'http://kdd.ics.uci.edu/databases/eeg/eeg_full.tar');
    fprintf('Extracting files...\n');
    untar('../Data/eeg_full.tar', '../Data/eeg_full/');
    system('chmod 755 ../Data/eeg_full/*.tar.gz');
    system('find ../Data/eeg_full/ -name "*.tar.gz" -exec tar -C ../Data/eeg_full/ -xf {} \; -exec rm -f {} \;');
    system('find ../Data/eeg_full/ -name "*.gz" -exec gunzip {} \;');
end

fprintf('Preprosessing the dataset...\n');
DATA_ROOT = '../Data/eeg_full/';
OUTPUT_DIR = '../Data/';
dataset_name = 'eeg';

root_folder = dir(DATA_ROOT);
sub_folders = root_folder([root_folder(:).isdir]);
sub_folders = sub_folders(~ismember({sub_folders(:).name},{'.','..'}));

num_samples = 0;
for i = 1 : length(sub_folders)
    data_files = dir([DATA_ROOT, sub_folders(i).name, '/co*']);
    num_samples = num_samples + length(data_files);
end

Xtrain = zeros([256, 64, num_samples]);
ytrain = zeros(num_samples, 1);

unwanted_lines = [1:4, 5:257:(5+257*63), 16453];
wanted_lines = setdiff(1:16453, unwanted_lines);

count = 0;
for i = 1 : length(sub_folders)
    data_files = dir([DATA_ROOT, sub_folders(i).name, '/co*']);
    if contains(sub_folders(i).name, 'co2a') || contains(sub_folders(i).name, 'co3a')
        curr_label = 1;
    elseif contains(sub_folders(i).name, 'co2c') || contains(sub_folders(i).name, 'co3c')
        curr_label = -1;
    else
        error('unknown class');
    end
    for j = 1 : length(data_files)
        curr_file_name = [DATA_ROOT, sub_folders(i).name, '/', data_files(j).name];
        curr_text_cell_arr = regexp(fileread(curr_file_name), '\n', 'split');
        if length(curr_text_cell_arr) ~= 16453
            fprintf('invalid file: %s\n', curr_file_name);
            continue;
        end
        count = count + 1;
        curr_text_cell_arr = curr_text_cell_arr(wanted_lines); 
        curr_text_arr = strjoin(curr_text_cell_arr);
        curr_text_split = strsplit(curr_text_arr);
        curr_text_split = curr_text_split(4:4:end);
        S = sprintf('%s ', curr_text_split{:});
        data_vec = sscanf(S, '%f');
        Xtrain(:, :, count) = reshape(data_vec, [256, 64]);
        ytrain(count) = curr_label;
    end
end

if count < num_samples
    Xtrain(:, :, count + 1 : num_samples) = [];
    ytrain(count + 1 : num_samples) = [];
end

Xtest = [];
ytest = [];

% sort data points 
[ytrain_new, new_idx] = sort(ytrain, 'descend');
Xtrain_new = Xtrain(:, :, new_idx);
Xtrain = Xtrain_new;
ytrain = ytrain_new;
% clip
Xtrain = Xtrain(:, :, 1:11000);
ytrain = ytrain(1:11000);

data_file_name = [OUTPUT_DIR, dataset_name, '_dataset.mat'];
save(data_file_name, 'Xtrain', 'ytrain', 'Xtest', 'ytest');
