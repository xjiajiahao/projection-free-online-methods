dataset = 'toy_stoch';

%% Fix the seed for reproducability
rng(0,'twister'); % To make sure we use the same data for all tests

%% Generate data
data_size = 1e3;
num_blocks = 5;
indBlocks = [0,sort(randperm(data_size,num_blocks-1)),data_size];
for t = 1:num_blocks
    v(indBlocks(t)+1:indBlocks(t+1),t) = 10*(rand(indBlocks(t+1)-indBlocks(t),1)-0.5);
end
model_covariance = v*v';
rng('shuffle')

model_radius = trace(model_covariance);
model_lambda = 0;

dim = size(v,1);
[uu, zz, ~] = svd(v);
zz = sparse(zz);
zz = sqrt(zz*zz');
sqrtSU = zz * sparse(uu');
clearvars zz uu

IS_NON_SMOOTH = false;
IS_CALCULATING_REGRET = true;
stoptime = inf;
num_iters_base = 400;

IS_SPARSE = false;

selected_methods = {};
selected_methods{end + 1} = 'OFWRG';
selected_methods{end + 1} = 'OSFW';
% selected_methods{end + 1} = 'OFW';
% selected_methods{end + 1} = 'OAW';


batch_size = 1e2;
sub_batch_size = 4;
print_freq = 1e0;

% OFWRG
eta_coef_OFWRG = 1e-0;
eta_exp_OFWRG = 1;
rho_coef_OFWRG = 1e-0;
rho_exp_OFWRG = 1;
beta0_OFWRG = 0;

% OSFW
eta_coef_OSFW = 5e-1;
eta_exp_OSFW = 1;
rho_coef_OSFW = 1e-0;
rho_exp_OSFW = 2/3;
beta0_OSFW = 0;

% OFW
eta_coef_OFW = 5e-1;
eta_exp_OFW = 1;
beta0_OFW = 0;

% OAW
eta_coef_OAW = 5e-1;
eta_exp_OAW = 1;
beta0_OAW = 0;
