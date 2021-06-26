dataset = 'Jester1';  % num_rows, num_cols, Xtrain, ytrain, Xtest, ytest
model_radius = 1e4;
model_lambda = 1e-3;

batch_size = 1e2;
sub_batch_size = 64;

print_freq = 1e0;
num_iters_base = 3e3;
IS_CALCULATING_REGRET = true;
stoptime = inf;


selected_methods = {};
selected_methods{end + 1} = 'OFWRG';
selected_methods{end + 1} = 'OSFW';
selected_methods{end + 1} = 'P_MOLES';
selected_methods{end + 1} = 'PD_MOLES';

IS_NON_SMOOTH = true;
IS_SPARSE = true;
IS_ADVERSARIAL = false;

% OFWRG
eta_coef_OFWRG = 2e-1;
eta_exp_OFWRG = 1e0;
rho_coef_OFWRG = 1e0;
rho_exp_OFWRG = 1e0;
beta0_OFWRG = 1e7;

% OSFW
eta_coef_OSFW = 2e-1;
eta_exp_OSFW = 1e0;
rho_coef_OSFW = 1e0;
rho_exp_OSFW = 2/3;
beta0_OSFW = 1e8;

% P_MOLES
eta_coef_P_MOLES = 1e5;
eta_exp_P_MOLES = 0;
lambda_P_MOLES = 1e7;
gamma_coef_P_MOLES = 1e0;

% PD_MOLES
eta_coef_PD_MOLES = 1e5;
eta_exp_PD_MOLES = 3/2;
lambda_PD_MOLES = 1e6;
gamma_coef_PD_MOLES = 1e-3;
