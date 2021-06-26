% if ~exist('../Data','dir'), mkdir Data; end

%% download and generate MovieLens100K dataset
fprintf('Downloading the MovieLens 100K dataset...\n');
if ~exist('../Data/ml-100k/u.data', 'file')
    websave('../Data/ml-100k.zip', 'http://files.grouplens.org/datasets/movielens/ml-100k.zip');
    unzip('../Data/ml-100k.zip','../Data/');
end

fprintf('Preprosessing...\n');
if ~exist('../Data/MovieLens100K.mat', 'file')
   gen_data_MovieLens100K;
   clear;
end

if ~exist('../Data/MovieLens100K_dataset.mat', 'file')
    gen_CF_dataset('MovieLens100K', 0.8);
    clear;
end

%% download and generate Jester1 dataset
fprintf('Downloading the Jester 1 dataset...\n');
if ~exist('../Data/jester-data-1.xls', 'file')
    websave('../Data/jester_dataset_1_1.zip', 'http://eigentaste.berkeley.edu/dataset/jester_dataset_1_1.zip');
    unzip('../Data/jester_dataset_1_1.zip','../Data/');
end

fprintf('Preprosessing...\n');
if ~exist('../Data/Jester1.mat', 'file')
   gen_data_Jester('jester-data-1.xls', 'Jester1');
   clear;
end

if ~exist('../Data/Jester1_dataset.mat', 'file')
    gen_CF_dataset('Jester1', 0.8);
    clear;
end

%% download and generate MovieLens1M dataset
fprintf('Downloading the MovieLens 1M dataset...\n');
if ~exist('../Data/ml-1m/ratings.dat', 'file')
    websave('../Data/ml-1m.zip', 'http://files.grouplens.org/datasets/movielens/ml-1m.zip');
    unzip('../Data/ml-1m.zip','../Data/');
end

fprintf('Preprosessing...\n');
if ~exist('../Data/MovieLens1M.mat', 'file')
   gen_data_MovieLens1M;
   clear;
end

if ~exist('../Data/MovieLens1M_dataset.mat', 'file')
    gen_CF_dataset('MovieLens1M', 0.8);
    clear;
end

%% download and generate Jester3 dataset
fprintf('Downloading the Jester 3 dataset...\n');
if ~exist('../Data/FINAL jester 2006-15.xls', 'file')
    websave('../Data/JesterDataset3.zip', 'http://eigentaste.berkeley.edu/dataset/JesterDataset3.zip');
    unzip('../Data/JesterDataset3.zip','../Data/');
end

fprintf('Preprosessing...\n');
if ~exist('../Data/Jester3.mat', 'file')
   gen_data_Jester('FINAL jester 2006-15.xls', 'Jester3');
   clear;
end

if ~exist('../Data/Jester3_dataset.mat', 'file')
    gen_CF_dataset('Jester3', 0.8);
    clear;
end


function gen_CF_dataset(data_name, ratio_observation)
    DATA_ROOT = '../Data/';

    filename = [DATA_ROOT, data_name, '.mat'];
    load(filename);

    rng(1);  % for reproducibility

    % parameters
    [num_rows, num_cols] = size(user_ratings_matrix);

    nnz_ratings = nnz(user_ratings_matrix);
    non_zero_indices = find(user_ratings_matrix)';

    num_observed_entries = ceil(nnz_ratings * ratio_observation);
    indices_entries = randperm(nnz_ratings);
    non_zero_indices = non_zero_indices(indices_entries);


    % training set
    indices_observed_entries = non_zero_indices(1 : num_observed_entries);
    values_observed_entries = user_ratings_matrix(indices_observed_entries);
    Xtrain = indices_observed_entries;
    ytrain = full(values_observed_entries);

    % test set
    indices_test_entries = non_zero_indices(num_observed_entries + 1 : end);
    values_test_entries = user_ratings_matrix(indices_test_entries);
    Xtest = indices_test_entries;
    ytest = full(values_test_entries);

    data_file_name = [DATA_ROOT, data_name, '_dataset.mat'];
    save(data_file_name, 'num_rows', 'num_cols', 'Xtrain', 'ytrain', 'Xtest', 'ytest');
end
