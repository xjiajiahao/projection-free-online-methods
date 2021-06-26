function gen_data_Jester(excel_file_name, dataset_name)
    ROOT = '../Data/';
    OUTPUT_DIR = '../Data/';
    
    % load the excel table
    file_name = [ROOT, excel_file_name];
    user_ratings_matrix = transpose(table2array(readtable(file_name, 'ReadVariableNames', false)));  % (num_movies+1)-by-num_users
    % preprocessing
    user_ratings_matrix = user_ratings_matrix(2:end, :);  % remove the first row, which is the number of non-zero ratings of each user
    num_movies = size(user_ratings_matrix, 1);
    num_users = size(user_ratings_matrix, 2);
    
    % % rescale and sparsify the matrix
    % user_ratings_matrix = 0.2 * user_ratings_matrix + 3.0;   % rescale from [-10, 10] to [1, 5]
    % user_ratings_matrix(user_ratings_matrix == 0.2 * 99.0 + 3.0) = 0.0;
    % user_ratings_matrix(isnan(user_ratings_matrix)) = 0.0;  % replace NAN with 0
    %
    % rescale and sparsify the matrix
    valid_entries = user_ratings_matrix(find(user_ratings_matrix ~= 99.0));
    invalid_indices = find(user_ratings_matrix == 99.0);
    max_value = max(valid_entries);
    min_value = min(valid_entries);
    user_ratings_matrix = 1.0 + (user_ratings_matrix - min_value) ./ (max_value - min_value) * 4;   % rescale from [-10, 10] to [1, 5]
    user_ratings_matrix(invalid_indices) = 0.0;
    user_ratings_matrix(isnan(user_ratings_matrix)) = 0.0;  % replace NAN with 0

    user_ratings_matrix = sparse(user_ratings_matrix);
    
    filename = [OUTPUT_DIR, dataset_name, '.mat'];
    save(filename, 'user_ratings_matrix');
end  % end of the function
