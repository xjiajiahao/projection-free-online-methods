function gen_data_MovieLens100K()
%% initialization
ROOT = '../Data/ml-100k/';
OUTPUT_DIR = '../Data/';
num_movies = 1682;
num_users = 943;

% create a movie_id_dict mapping from old movie id to new movie id
movie_id_dict = py.dict();

% allocate a ratings matrix for each user
user_ratings_cell = cell(1, num_users);
user_ratings_matrix = zeros(num_movies, num_users);

%% Process movies.dat file
f_movies = fopen([ROOT, 'u.item'], 'r');
counter = 0;

while ~feof(f_movies)
    % Step 1: read one line
    counter = counter + 1;
    line = fgetl(f_movies); %# read line by line
    fields = strsplit(line, '|');

    % Step 2: remap the movie_id
    old_movie_id_str = fields{1};
    update(movie_id_dict, py.dict(pyargs(old_movie_id_str, counter)));
end

%% Process ratings.dat file
f_ratings= fopen([ROOT, 'u.data'], 'r');
while ~feof(f_ratings)
    % Step 1: read one line
    line = fgetl(f_ratings); %# read line by line
    fields = strsplit(line);
    user_id = str2num(fields{1});
    old_movie_id_str = fields{2};
    score = str2num(fields{3});
    % Step 2: build user_ratings_cell
    new_movie_id = movie_id_dict{old_movie_id_str};
    user_ratings_cell{user_id} = [user_ratings_cell{user_id}; new_movie_id, score];
end

% sort the movie ratings from high to low
for i = 1 : num_users
    tmp_matrix = user_ratings_cell{i};
    for j = 1 : size(tmp_matrix, 1)
        user_ratings_matrix(tmp_matrix(j, 1), i) = tmp_matrix(j, 2);
    end
    user_ratings_cell{i} = sortrows(user_ratings_cell{i}, 2, 'descend');
    user_ratings_cell{i} = user_ratings_cell{i}';
end

user_ratings_matrix = sparse(user_ratings_matrix);


filename = [OUTPUT_DIR, 'MovieLens100K', '.mat'];
save(filename, 'user_ratings_cell', 'user_ratings_matrix');
