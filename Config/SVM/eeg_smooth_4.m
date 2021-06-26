dataset = 'eeg';  % num_rows, num_cols, Xtrain, ytrain, Xtest, ytest
pos_label = 1;
model_alpha = 3;
model_radius = 1e0;
model_lambda = 0;

batch_size = 440;
sub_batch_size = 4;

print_freq = 1e0;
% num_iters_base = 1e2;
num_iters_base = 1e1;
IS_CALCULATING_REGRET = true;
stoptime = inf;


selected_methods = {};
selected_methods{end + 1} = 'Meta_OFWRG';
selected_methods{end + 1} = 'PFOCO_SG';
selected_methods{end + 1} = 'Meta_FW';

IS_NON_SMOOTH = false;
IS_SPARSE = true;
IS_ADVERSARIAL = true;

% ROFW
eta_coef_ROFW = 2e-3;
eta_exp_ROFW = 1/4;
reg_coef_ROFW = 1e7;

% OSPF
perturbation_OSPF = 1e2;

% Meta_OFWRG
eta_coef_Meta_OFWRG = 1e-3;
eta_exp_Meta_OFWRG = 1e0;
rho_coef_Meta_OFWRG = 1e0;
rho_exp_Meta_OFWRG = 1/2;
reg_coef_Meta_OFWRG = 1e6;
beta0_Meta_OFWRG = 0;

% Meta_FW
eta_coef_Meta_FW = 1e-3;
eta_exp_Meta_FW = 1e0;
rho_coef_Meta_FW = 1e0;
rho_exp_Meta_FW = 2/3;
reg_coef_Meta_FW = 1e6;
beta0_Meta_FW = 0;

% PFOCO_SG
eta_coef_PFOCO_SG = 1e-3;
eta_exp_PFOCO_SG = 1e0;
reg_coef_PFOCO_SG = 1e8;
beta0_PFOCO_SG = 0;

% SFPL
perturbation_SFPL = 1e2;
