dataset = 'toy_adv';

num_iters_base = 400;
sub_batch_size = 16;
print_freq = 1e0;

model_radius = 1.1681e+04;
model_lambda = 0;

IS_CALCULATING_REGRET = true;
IS_ADVERSARIAL = true;
IS_NON_SMOOTH = false;
IS_SPARSE = false;
stoptime = inf;

selected_methods = {};
selected_methods{end + 1} = 'Meta_OFWRG';
selected_methods{end + 1} = 'PFOCO_SG';


%% Fix the seed for reproducability
rng(0,'twister'); % To make sure we use the same data for all tests

%% Generate data
dim = 1e3;
num_blocks = 5;
sqrtSU_collections = cell(1, num_blocks);
blocks_original = zeros(dim, num_blocks);
indBlocks = [0,sort(randperm(dim, num_blocks - 1)), dim];
for t = 1 : num_blocks
    blocks_original(indBlocks(t)+1:indBlocks(t+1),t) = 10*(rand(indBlocks(t+1)-indBlocks(t),1)-0.5);
end

v = blocks_original;
[uu, zz, ~] = svd(v);
zz = sparse(zz);
zz = sqrt(zz*zz');
tmp_sqrtSU = zz * sparse(uu');

for t = 1 : num_iters_base
    rot_main_diag = sparse(1:dim, 1:dim, ones(dim, 1) .* cos(10 * pi * t / num_iters_base), dim, dim);
    rot_upper_diag = sparse(1:dim-1, 2:dim, ones(dim - 1, 1) .* sin(10 * pi * t / num_iters_base), dim, dim);
    rot_lower_diag = sparse(2:dim, 1:dim-1, ones(dim - 1, 1) .* (-sin(10 * pi * t / num_iters_base)), dim, dim);
    rot_mat = rot_main_diag + rot_upper_diag + rot_lower_diag;
    sqrtSU_collections{t} = tmp_sqrtSU * rot_mat;
end

rng('shuffle')



% MOFWRG
eta_coef_Meta_OFWRG = 1e-0;
eta_exp_Meta_OFWRG = 1e0;
rho_coef_Meta_OFWRG = 1e0;
rho_exp_Meta_OFWRG = 1;
reg_coef_Meta_OFWRG = 1e0;
beta0_Meta_OFWRG = 0;

% PFOCO_SG
eta_coef_PFOCO_SG = 1e-0;
eta_exp_PFOCO_SG = 1e0;
reg_coef_PFOCO_SG = 1e0;
beta0_PFOCO_SG = 0;
