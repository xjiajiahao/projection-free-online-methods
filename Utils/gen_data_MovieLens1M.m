function gen_data_MovieLens1M()
%% initialization
ROOT = '../Data/ml-1m/';
OUTPUT_DIR = '../Data/';
num_movies = 3883;
num_users = 6040;
num_genres = 18;

% create a genres_dict mapping from genre name to number
genres_dict = py.dict(pyargs( ...
    'Action', 1, ...
	'Adventure', 2, ...
	'Animation', 3, ...
	'Children''s', 4, ...
	'Comedy', 5, ...
	'Crime', 6, ...
	'Documentary', 7, ...
	'Drama', 8, ...
	'Fantasy', 9, ...
	'Film-Noir', 10, ...
	'Horror', 11, ...
	'Musical', 12, ...
	'Mystery', 13, ...
	'Romance', 14, ...
	'Sci-Fi', 15, ...
	'Thriller', 16, ...
	'War', 17, ...
	'Western', 18));
% create a matrix that records the movie-genres information
movie_genre = zeros(num_movies, num_genres);

% create a movie_id_dict mapping from old movie id to new movie id
movie_id_dict = py.dict();

% allocate a ratings matrix for each user
user_ratings_cell = cell(1, num_users);
user_ratings_matrix = zeros(num_movies, num_users);

%% Process movies.dat file
f_movies = fopen([ROOT, 'movies.dat'], 'r');
counter = 0;

while ~feof(f_movies)
    % Step 1: read one line
    counter = counter + 1;
    line = fgetl(f_movies); % read line by line
    fields = strsplit(line, '::');

    % Step 2: remap the movie_id
    old_movie_id_str = fields{1};
    update(movie_id_dict, py.dict(pyargs(old_movie_id_str, counter)));

    % Step 3: record current movie's genres
    cur_genres = strsplit(fields{end}, '|');
    cur_genres_num = size(cur_genres, 2);
    for i = 1 : cur_genres_num
        tmp_genre = genres_dict{cur_genres{i}};
        movie_genre(counter, tmp_genre) = 1;
    end
end

movie_genre = sparse(movie_genre);

%% Process ratings.dat file
f_ratings = fopen([ROOT, 'ratings.dat'], 'r');

while ~feof(f_ratings)
    % Step 1: read one line
    line = fgetl(f_ratings); %# read line by line
    fields = strsplit(line, '::');
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

filename = [OUTPUT_DIR, 'MovieLens1M', '.mat'];
save(filename, 'user_ratings_cell', 'user_ratings_matrix');
