%% Load data
addpath('./Algorithms/vector/');
addpath('./Config/CF/');
addpath('./Utils/');
DATA_ROOT = './Data/';

% Set parameters (feel free to modify parameters in this section)
if ~exist('IS_TUNING_PARAMETERS', 'var') || IS_TUNING_PARAMETERS == false
    % MovieLens1M_64;

    data_file_name = [DATA_ROOT, dataset, '_dataset.mat'];
    load(data_file_name, 'Xtrain', 'ytrain', 'Xtest', 'ytest', 'num_rows', 'num_cols');
end

[~, data_size] = size(Xtrain);

optimal_value_file_name = [DATA_ROOT, dataset, '_CF_opt.mat'];
optimal_solution = [];
if exist(optimal_value_file_name, 'file') == 2
    load(optimal_value_file_name, 'optimal_value', 'optimal_solution');
    IS_OPTIMAL_VALUE_AVAILABLE = true;
else
    IS_OPTIMAL_VALUE_AVAILABLE = false;
end

if ~IS_OPTIMAL_VALUE_AVAILABLE && IS_CALCULATING_REGRET
    IS_CALCULATING_REGRET = false;
end

%% Run algorithms, do NOT modify the code below
num_iters_OFWRG = num_iters_base;
num_iters_OSFW = num_iters_base;
num_iters_P_MOLES = num_iters_base;
num_iters_PD_MOLES = num_iters_base;
num_iters_FW = num_iters_base;

W0 = zeros(num_rows, num_cols);

loss_handle = @(W, X, y) loss_fn(W, X, y, model_lambda);
grad_handle = @grad_fn;
subgrad_handle = @(W, X, y) subgrad_fn(W, X, y, model_lambda);
lmo_handle = @(V) lmo_trace_norm_fn(V, model_radius);
gap_handle = @(W, X, y) gap_fn(W, X, y);
grad_moreau_smoothed_reg_handle = @(W, beta) grad_moreau_smoothed_reg_fn(W, beta, model_lambda);
ball_radius = get_F_norm_ball_radius(model_radius);

obj_values_cell = cell(length(selected_methods), 1);
for method_idx = 1 : length(selected_methods)
    curr_method = selected_methods{method_idx};
    if strcmp(curr_method, 'OFWRG') == true  % OFWRG
        if IS_NON_SMOOTH
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_OFWRG), ', \eta exp=', num2str(eta_exp_OFWRG), ', \rho coef=', num2str(rho_coef_OFWRG), ', \rho exp=', num2str(rho_exp_OFWRG), ', \beta=', num2str(beta0_OFWRG), ', batch=', num2str(sub_batch_size)];
        else
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_OFWRG), ', \eta exp=', num2str(eta_exp_OFWRG), ', \rho coef=', num2str(rho_coef_OFWRG), ', \rho exp=', num2str(rho_exp_OFWRG), ', batch=', num2str(sub_batch_size)];
        end
        fprintf('%s\n', curr_label);
        [solution, obj_values] = OFWRG(W0, Xtrain, ytrain, Xtest, ytest, num_iters_OFWRG, batch_size, sub_batch_size, eta_coef_OFWRG, eta_exp_OFWRG, rho_coef_OFWRG, rho_exp_OFWRG, loss_handle, grad_handle, lmo_handle, gap_handle, print_freq, IS_CALCULATING_REGRET, optimal_value, beta0_OFWRG, grad_moreau_smoothed_reg_handle, stoptime);
    elseif strcmp(curr_method, 'OSFW') == true  % OSFW
        if IS_NON_SMOOTH
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_OSFW), ', \eta exp=', num2str(eta_exp_OSFW), ', \rho coef=', num2str(rho_coef_OSFW), ', \rho exp=', num2str(rho_exp_OSFW), ', \beta=', num2str(beta0_OSFW), ', batch=', num2str(sub_batch_size)];
        else
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_OSFW), ', \eta exp=', num2str(eta_exp_OSFW), ', \rho coef=', num2str(rho_coef_OSFW), ', \rho exp=', num2str(rho_exp_OSFW), ', batch=', num2str(sub_batch_size)];
        end
        fprintf('%s\n', curr_label);
        [solution, obj_values] = OSFW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_OSFW, batch_size, sub_batch_size, eta_coef_OSFW, eta_exp_OSFW, rho_coef_OSFW, rho_exp_OSFW, loss_handle, grad_handle, lmo_handle, gap_handle, print_freq, IS_CALCULATING_REGRET, optimal_value, beta0_OSFW, grad_moreau_smoothed_reg_handle, stoptime);
    elseif strcmp(curr_method, 'FW') == true  % Full FW
        if IS_NON_SMOOTH
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_FW), ', \eta exp=', num2str(eta_exp_FW), ', \beta=', num2str(beta0_FW)];
        else
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_FW), ', \eta exp=', num2str(eta_exp_FW)];
        end
        fprintf('%s\n', curr_label);
        [solution, obj_values] = FW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_FW, eta_coef_FW, eta_exp_FW, loss_handle, grad_handle, lmo_handle, gap_handle, beta0_FW, grad_moreau_smoothed_reg_handle);
    elseif strcmp(curr_method, 'P_MOLES') == true  % P_MOLES
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_P_MOLES), ', \eta exp=', num2str(eta_exp_P_MOLES), ', lambda=', num2str(lambda_P_MOLES), ', \gamma coef=', num2str(gamma_coef_P_MOLES), ', batch=', num2str(sub_batch_size)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = P_MOLES(W0, Xtrain, ytrain, Xtest, ytest, num_iters_P_MOLES, batch_size, sub_batch_size, eta_coef_P_MOLES, eta_exp_P_MOLES, lambda_P_MOLES, gamma_coef_P_MOLES, ball_radius, loss_handle, subgrad_handle, lmo_handle, gap_handle, print_freq, IS_CALCULATING_REGRET, optimal_value, stoptime);
    elseif strcmp(curr_method, 'PD_MOLES') == true  % PD_MOLES
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_PD_MOLES), ', \eta exp=', num2str(eta_exp_PD_MOLES), ', lambda=', num2str(lambda_PD_MOLES), ', \gamma coef=', num2str(gamma_coef_PD_MOLES), ', batch=', num2str(sub_batch_size)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = PD_MOLES(W0, Xtrain, ytrain, Xtest, ytest, num_iters_PD_MOLES, batch_size, sub_batch_size, eta_coef_PD_MOLES, eta_exp_PD_MOLES, lambda_PD_MOLES, gamma_coef_PD_MOLES, ball_radius, loss_handle, subgrad_handle, lmo_handle, gap_handle, print_freq, IS_CALCULATING_REGRET, optimal_value, stoptime);
    else
        error('method name mismatched');
    end

    % plot curves
    if IS_OPTIMAL_VALUE_AVAILABLE
        if IS_CALCULATING_REGRET
            plot(obj_values(:, 1), cumsum(obj_values(:, 4)), 'DisplayName', curr_label); hold on;
            xlabel('#iterations');
            ylabel('regret');
            legend('show', 'Location', 'northwest');
        else
            plot(obj_values(:, 1), obj_values(:, 4), 'DisplayName', curr_label); hold on;
            xlabel('#rounds');
            ylabel('loss');
            legend('show');
        end
    else
        plot(obj_values(:, 1), obj_values(:, 4), 'DisplayName', curr_label); hold on;
        xlabel('#iterations');
        ylabel('loss value');
        legend('show');
    end
    obj_values_cell{method_idx} = obj_values;

    % training loss
    training_loss = loss_handle(solution, Xtrain, ytrain);
    fprintf('training loss: %f\n', training_loss);
end
drawnow;

%% Definitions of loss function, gradient, and linear optimization oracle
output_file_name = [DATA_ROOT, 'results_', dataset, '_CF_auto_save.mat'];
save(output_file_name, 'selected_methods', 'obj_values_cell');
beep;

grid on;

function loss = loss_fn(W, X, y, lambda)
    % square loss
    diff = W(X) - y;
    loss = sum(diff.^2) / (2 * length(y)) + lambda * (max(max(abs(W))));
end

function grad = grad_fn(W, X, y)
    data_size = size(X, 2);
    [num_rows, ~] = size(W);
    if data_size < numel(W) * 0.5
        diff = (W(X) - y) ./ (length(y));
        grad = sparse(mod(X - 1, num_rows) + 1, ceil(X ./ num_rows), diff, size(W, 1), size(W, 2));
    elseif data_size < numel(W)
        diff = (W(X) - y) ./ (length(y));
        grad = zeros(size(W));
        grad(X) = diff;
    else
        grad = (W - reshape(y, size(W))) ./ (length(y));
    end
end

function subgrad = subgrad_fn(W, X, y, lambda)
    data_size = size(X, 2);
    [num_rows, ~] = size(W);
    % first, find the largest magnitude in W
    [~, max_idx_flat] = max(abs(W(:) - 3));
    max_value_sign = sign(W(max_idx_flat) - 3);
    if data_size < numel(W) * 0.5
        diff = (W(X) - y) ./ (length(y));
        max_idx_flat_in_X =  find(X == max_idx_flat);
        if ~isempty(max_idx_flat_in_X)
            diff(max_idx_flat_in_X(1)) = diff(max_idx_flat_in_X(1)) + lambda * max_value_sign;
            subgrad = sparse(mod(X - 1, num_rows) + 1, ceil(X ./ num_rows), diff, size(W, 1), size(W, 2));
        else
            subgrad = sparse(mod([X, max_idx_flat] - 1, num_rows) + 1, ceil([X, max_idx_flat] ./ num_rows), [diff, lambda * max_value_sign], size(W, 1), size(W, 2));
        end
    elseif data_size < numel(W)
        diff = (W(X) - y) ./ (length(y));
        subgrad = zeros(size(W));
        subgrad(X) = diff;
        subgrad(max_idx_flat) = subgrad(max_idx_flat) + lambda * max_value_sign;
    else
        subgrad = (W - reshape(y, size(W))) ./ (length(y));
        subgrad(max_idx_flat) = subgrad(max_idx_flat) + lambda * max_value_sign;
    end
end

function res = lmo_trace_norm_fn(A, radius)  % lmo for nuclear norm constraint || X ||_* <= radius
    [lvec, ~, rvec] = bksvd(A, 1);
    res = lvec*(-radius)*rvec';
end

function fw_gap = gap_fn(W, X, y)
    % @NOTE no need to compute the Frank-Wolfe gap, simply set to zero
    fw_gap = 0;
end

function ret = grad_moreau_smoothed_reg_fn(W, beta, lambda)
    ret = zeros(numel(W), 1);
    proj_l1(W(:) - 3, ret, beta * lambda);
    ret = reshape(ret, size(W)) .* lambda;
end

function ret = get_F_norm_ball_radius(model_radius)
    ret = model_radius;
end
