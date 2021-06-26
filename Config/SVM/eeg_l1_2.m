dataset = 'eeg';
pos_label = 1;
model_alpha = 3;
model_radius = 1e0;
model_lambda = 1e-3;

batch_size = 440;
sub_batch_size = 2;

print_freq = 1e0;
% num_iters_base = 1e2;
num_iters_base = 1e1;
IS_CALCULATING_REGRET = true;
stoptime = inf;


selected_methods = {};
selected_methods{end + 1} = 'Meta_OFWRG';

IS_NON_SMOOTH = true;
IS_SPARSE = true;
IS_ADVERSARIAL = true;

% ROFW
eta_coef_ROFW = 2e-4;
eta_exp_ROFW = 1/4;
reg_coef_ROFW = 1e8;

% OSPF
perturbation_OSPF = 1e3;

% Meta_OFWRG
eta_coef_Meta_OFWRG = 5e-4;
eta_exp_Meta_OFWRG = 1e0;
rho_coef_Meta_OFWRG = 1e0;
rho_exp_Meta_OFWRG = 1;
reg_coef_Meta_OFWRG = 1e4;
beta0_Meta_OFWRG = 1e1;

% SFPL
perturbation_SFPL = 1e1;
