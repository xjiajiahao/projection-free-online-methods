% clear;
%% Load data
addpath('./Algorithms/vector_sample/');
addpath('./Config/CME/');
DATA_ROOT = './Data/';

% Set parameters (feel free to modify parameters in this section)
if ~exist('IS_TUNING_PARAMETERS', 'var') || IS_TUNING_PARAMETERS == false
    % toy_stoch_64;
end

optimal_solution = [];
optimal_value = 0.0;
IS_OPTIMAL_VALUE_AVAILABLE = true;

%% Run algorithms
num_iters_OFWRG = num_iters_base;
num_iters_OSFW = num_iters_base;
num_iters_OFW = num_iters_base;
num_iters_OAW = num_iters_base;
num_iters_FW = num_iters_base;

W0 = zeros(dim);

loss_handle = @(X) loss_fn(X, model_covariance);
grad_handle = @grad_fn;
subgrad_handle = @(X, samples) subgrad_fn(X, samples);
exact_grad_handle = @(X) exact_grad_fn(X, model_covariance);
sample_handle = @(num_samples) generate_sample(num_samples, dim, sqrtSU);
lmo_handle = @(V) lmo_trace_norm_fn(V, model_radius);
grad_moreau_smoothed_reg_handle = @grad_moreau_smoothed_reg_fn;  % not used
gap_handle = @(X) gap_fn(X, model_radius);
ball_radius = get_F_norm_ball_radius(model_radius);


obj_values_cell = cell(length(selected_methods), 1);
solutions_cell = cell(length(selected_methods), 1);

for method_idx = 1 : length(selected_methods)
    curr_method = selected_methods{method_idx};
    if strcmp(curr_method, 'OFWRG') == true  % OFWRG
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_OFWRG), ', \eta exp=', num2str(eta_exp_OFWRG), ', \rho coef=', num2str(rho_coef_OFWRG), ', \rho exp=', num2str(rho_exp_OFWRG), ', \beta =', num2str(beta0_OFWRG), ', batch=', num2str(sub_batch_size)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = OFWRG(W0, sample_handle, num_iters_OFWRG, batch_size, sub_batch_size, eta_coef_OFWRG, eta_exp_OFWRG, rho_coef_OFWRG, rho_exp_OFWRG, beta0_OFWRG, loss_handle, grad_handle, lmo_handle, gap_handle, grad_moreau_smoothed_reg_handle, print_freq, IS_CALCULATING_REGRET, optimal_value, stoptime);
    elseif strcmp(curr_method, 'OSFW') == true  % OSFW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_OSFW), ', \eta exp=', num2str(eta_exp_OSFW), ', \rho coef=', num2str(rho_coef_OSFW), ', \rho exp=', num2str(rho_exp_OSFW), ', \beta =', num2str(beta0_OSFW), ', batch=', num2str(sub_batch_size)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = OSFW(W0, sample_handle, num_iters_OSFW, batch_size, sub_batch_size, eta_coef_OSFW, eta_exp_OSFW, rho_coef_OSFW, rho_exp_OSFW, beta0_OSFW, loss_handle, grad_handle, lmo_handle, gap_handle, grad_moreau_smoothed_reg_handle, print_freq, IS_CALCULATING_REGRET, optimal_value, stoptime, IS_SPARSE);
    elseif strcmp(curr_method, 'FW') == true  % FW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_FW), ', \eta exp=', num2str(eta_exp_FW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = FW(W0, num_iters_FW, batch_size, eta_coef_FW, eta_exp_FW, beta0_FW, loss_handle, exact_grad_handle, lmo_handle, gap_handle, grad_moreau_smoothed_reg_handle, print_freq, IS_CALCULATING_REGRET, stoptime);
    elseif strcmp(curr_method, 'OFW') == true  % OFW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_OFW), ', \eta exp=', num2str(eta_exp_OFW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = OFW(W0, sample_handle, num_iters_OFW, batch_size, eta_coef_OFW, eta_exp_OFW, loss_handle, grad_handle, lmo_handle, gap_handle, print_freq, IS_CALCULATING_REGRET, optimal_value, stoptime, IS_SPARSE);
    elseif strcmp(curr_method, 'OAW') == true  % OAW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_OAW), ', \eta exp=', num2str(eta_exp_OAW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = OAW(W0, sample_handle, num_iters_OAW, batch_size, eta_coef_OAW, eta_exp_OAW, loss_handle, grad_handle, lmo_handle, gap_handle, print_freq, IS_CALCULATING_REGRET, optimal_value, stoptime, IS_SPARSE);
    else
        error('method name mismatched');
    end

    % plot curves
    if IS_CALCULATING_REGRET
        plot(obj_values(:, 1), cumsum(obj_values(:, 4)), 'DisplayName', curr_label); hold on;
        xlabel('#iterations');
        ylabel('regret');
        legend('show', 'Location', 'northwest');
    else
        semilogy(obj_values(:, 3), (obj_values(:, 4) - optimal_value) ./ (obj_values(1, 4) - optimal_value), 'DisplayName', curr_label); hold on;
        xlabel('time (s)');
        ylabel('loss value');
        legend('show');
    end
    obj_values_cell{method_idx} = obj_values;
    solutions_cell{method_idx} = solution;

    % training loss
    training_loss = loss_handle(solution);
    fprintf('training loss: %f\n', training_loss);
end

%% Definitions of loss function, gradient, and linear optimization oracle
output_file_name = [DATA_ROOT, 'results_', dataset, '_CME_auto_save.mat'];
save(output_file_name, 'selected_methods', 'obj_values_cell', 'solutions_cell');
beep;

grid on;

% objective function: covariance matrix estiamtion
% f(X) = E_w ||X - w w^T||_F^2 , s.t., X \in S_+^d, ||X||_* <= radius
function loss = loss_fn(X, model_covariance)
    loss = norm(X - model_covariance,'fro')^2 / 2;
end

function ret = generate_sample(num_samples, dim, sqrtSU)
    ret = randn(num_samples, dim) * sqrtSU;
end

function grad = grad_fn(X, samples)
    num_samples = size(samples, 1);
    grad = X - ((samples ./ num_samples)' * samples);
end

function subgrad = subgrad_fn(X, samples)
    num_samples = size(samples, 1);
    subgrad = X - ((samples ./ num_samples)' * samples);
end

function exact_grad = exact_grad_fn(X, model_covariance)
    exact_grad = X - model_covariance;
end

function ret = lmo_trace_norm_fn(A, radius)  % lmo for nuclear norm constraint || X ||_* <= radius
    dim = size(A, 1);
    optsEigs.p = 5;
    optsEigs.tol = 1e-6;
    if dim <= 500
        [uLnczs,dLnczs] = eig(A);
        dLnczs = diag(dLnczs);
        [vaMin,inMin] = min(dLnczs);
        if vaMin < 0
            uLnczs = uLnczs(:,inMin);
            uLnczs = sqrt(radius)*uLnczs;
            ret = uLnczs*uLnczs';
        else
            ret = zeros(dim);
        end
    else
        [uLnczs,vaMin] = eigs(A, 1, 'SA', optsEigs);
        if vaMin < 0
            uLnczs = sqrt(radius) * uLnczs;
            ret = uLnczs * uLnczs';
        else
            ret = zeros(dim);
        end
    end
end

function ret = grad_moreau_smoothed_reg_fn(X, beta)
    ret = zeros(size(X));
end

function fw_gap = gap_fn(X, radius)
    % @NOTE no need to compute the Frank-Wolfe gap, simply set to zero
    fw_gap = 0;
end

function ret = get_F_norm_ball_radius(model_radius)
    ret = model_radius;
end
