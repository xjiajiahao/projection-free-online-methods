%% Load data
addpath('./Algorithms/vector/');
addpath('./Config/SVM/');
addpath('./Utils/');
DATA_ROOT = './Data/';

% Set parameters (feel free to modify parameters in this section)
if ~exist('IS_TUNING_PARAMETERS', 'var') || IS_TUNING_PARAMETERS == false
    % eeg_smooth_8;

    data_file_name = [DATA_ROOT, dataset, '_dataset.mat'];
    load(data_file_name, 'Xtrain', 'ytrain', 'Xtest', 'ytest');
end

[num_rows, num_cols, data_size] = size(Xtrain);

if IS_NON_SMOOTH
    optimal_value_file_name = [DATA_ROOT, dataset, '_SVM_l1_opt.mat'];
else
    optimal_value_file_name = [DATA_ROOT, dataset, '_SVM_smooth_opt.mat'];
end
optimal_solution = [];
optimal_value = [];
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
num_iters_Meta_OFWRG = num_iters_base;
num_iters_Meta_FW = num_iters_base;
num_iters_PFOCO_SG = num_iters_base;
num_iters_ROFW = num_iters_base;
num_iters_OSPF = num_iters_base;
num_iters_SFPL = num_iters_base;
num_iters_FW = num_iters_base;

W0 = zeros(num_rows, num_cols);

loss_handle = @(W, X, y) loss_fn(W, X, y, model_alpha, model_lambda);
grad_handle = @(W, X, y) grad_fn(W, X, y, model_alpha);
grad_dff_handle = @(W, W_old, rho, X, y) grad_diff_fn(W, W_old, rho, X, y, model_alpha);
subgrad_handle = @(W, X, y) subgrad_fn(W, X, y, model_alpha, model_lambda);
lmo_handle = @(V) lmo_trace_norm_fn(V, model_radius);
gap_handle = @(W, X, y) gap_fn(W, X, y);
grad_moreau_smoothed_reg_handle = @(W, beta) grad_moreau_smoothed_reg_fn(W, beta, model_lambda);
ball_radius = get_F_norm_ball_radius(model_radius);

obj_values_cell = cell(length(selected_methods), 1);
for method_idx = 1 : length(selected_methods)
    curr_method = selected_methods{method_idx};
    if strcmp(curr_method, 'FW') == true  % offline FW
        if IS_NON_SMOOTH
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_FW), ', \eta exp=', num2str(eta_exp_FW), ', \beta=', num2str(beta0_FW)];
        else
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_FW), ', \eta exp=', num2str(eta_exp_FW)];
        end
        fprintf('%s\n', curr_label);
        [solution, obj_values] = FW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_FW, eta_coef_FW, eta_exp_FW, loss_handle, grad_handle, lmo_handle, gap_handle, beta0_FW, grad_moreau_smoothed_reg_handle);
    elseif strcmp(curr_method, 'ROFW') == true  % ROFW
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_ROFW), ', \eta exp=', num2str(eta_exp_ROFW), ', reg coef=', num2str(reg_coef_ROFW)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = ROFW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_ROFW, batch_size, eta_coef_ROFW, eta_exp_ROFW, reg_coef_ROFW, loss_handle, subgrad_handle, lmo_handle, gap_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, stoptime);
    elseif strcmp(curr_method, 'OSPF') == true  % OSPF
        curr_label = [curr_method, ', perturbation=', num2str(perturbation_OSPF)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = OSPF(W0, Xtrain, ytrain, Xtest, ytest, num_iters_OSPF, batch_size, perturbation_OSPF, IS_NON_SMOOTH, loss_handle, subgrad_handle, lmo_handle, gap_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, stoptime);
    elseif strcmp(curr_method, 'P_MOLES') == true  % P_MOLES
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_P_MOLES), ', \eta exp=', num2str(eta_exp_P_MOLES), ', lambda=', num2str(lambda_P_MOLES)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = P_MOLES(W0, Xtrain, ytrain, Xtest, ytest, num_iters_P_MOLES, batch_size, eta_coef_P_MOLES, eta_exp_P_MOLES, lambda_P_MOLES, ball_radius, loss_handle, subgrad_handle, lmo_handle, gap_handle, print_freq, IS_CALCULATING_REGRET, optimal_value, stoptime);
    elseif strcmp(curr_method, 'PD_MOLES') == true  % PD_MOLES
        curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_PD_MOLES), ', \eta exp=', num2str(eta_exp_PD_MOLES), ', lambda=', num2str(lambda_PD_MOLES)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = PD_MOLES(W0, Xtrain, ytrain, Xtest, ytest, num_iters_PD_MOLES, batch_size, eta_coef_PD_MOLES, eta_exp_PD_MOLES, lambda_PD_MOLES, ball_radius, loss_handle, subgrad_handle, lmo_handle, gap_handle, print_freq, IS_CALCULATING_REGRET, optimal_value, stoptime);
    elseif strcmp(curr_method, 'Meta_OFWRG') == true
        if IS_NON_SMOOTH
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_Meta_OFWRG), ', \eta exp=', num2str(eta_exp_Meta_OFWRG), ', \rho coef=', num2str(rho_coef_Meta_OFWRG), ', \rho exp=', num2str(rho_exp_Meta_OFWRG), ', reg coef=', num2str(reg_coef_Meta_OFWRG), ', \beta=', num2str(beta0_Meta_OFWRG), ', batch=', num2str(sub_batch_size)];
        else
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_Meta_OFWRG), ', \eta exp=', num2str(eta_exp_Meta_OFWRG), ', \rho coef=', num2str(rho_coef_Meta_OFWRG), ', \rho exp=', num2str(rho_exp_Meta_OFWRG), ', reg coef=', num2str(reg_coef_Meta_OFWRG), ', batch=', num2str(sub_batch_size)];
        end
        fprintf('%s\n', curr_label);
        [solution, obj_values] = Meta_OFWRG(W0, Xtrain, ytrain, Xtest, ytest, num_iters_Meta_OFWRG, batch_size, sub_batch_size, eta_coef_Meta_OFWRG, eta_exp_Meta_OFWRG, rho_coef_Meta_OFWRG, rho_exp_Meta_OFWRG, reg_coef_Meta_OFWRG, loss_handle, grad_handle, lmo_handle, gap_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, beta0_Meta_OFWRG, grad_moreau_smoothed_reg_handle, grad_dff_handle);
    elseif strcmp(curr_method, 'Meta_FW') == true
        if IS_NON_SMOOTH
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_Meta_FW), ', \eta exp=', num2str(eta_exp_Meta_FW), ', \rho coef=', num2str(rho_coef_Meta_FW), ', \rho exp=', num2str(rho_exp_Meta_FW), ', reg coef=', num2str(reg_coef_Meta_FW), ', \beta=', num2str(beta0_Meta_FW), ', batch=', num2str(sub_batch_size)];
        else
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_Meta_FW), ', \eta exp=', num2str(eta_exp_Meta_FW), ', \rho coef=', num2str(rho_coef_Meta_FW), ', \rho exp=', num2str(rho_exp_Meta_FW), ', reg coef=', num2str(reg_coef_Meta_FW), ', batch=', num2str(sub_batch_size)];
        end
        fprintf('%s\n', curr_label);
        [solution, obj_values] = Meta_FW(W0, Xtrain, ytrain, Xtest, ytest, num_iters_Meta_FW, batch_size, sub_batch_size, eta_coef_Meta_FW, eta_exp_Meta_FW, rho_coef_Meta_FW, rho_exp_Meta_FW, reg_coef_Meta_FW, loss_handle, grad_handle, lmo_handle, gap_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, beta0_Meta_FW, grad_moreau_smoothed_reg_handle);
    elseif strcmp(curr_method, 'PFOCO_SG') == true
        if IS_NON_SMOOTH
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_PFOCO_SG), ', \eta exp=', num2str(eta_exp_PFOCO_SG), ', reg coef=', num2str(reg_coef_PFOCO_SG), ', \beta=', num2str(beta0_PFOCO_SG), ', batch=', num2str(sub_batch_size)];
        else
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_PFOCO_SG), ', \eta exp=', num2str(eta_exp_PFOCO_SG), ', reg coef=', num2str(reg_coef_PFOCO_SG), ', batch=', num2str(sub_batch_size)];
        end
        fprintf('%s\n', curr_label);
        [solution, obj_values] = PFOCO_SG(W0, Xtrain, ytrain, Xtest, ytest, num_iters_PFOCO_SG, batch_size, sub_batch_size, eta_coef_PFOCO_SG, eta_exp_PFOCO_SG, reg_coef_PFOCO_SG, loss_handle, grad_handle, lmo_handle, gap_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, beta0_PFOCO_SG, grad_moreau_smoothed_reg_handle);
    elseif strcmp(curr_method, 'SFPL') == true  % SFPL
        curr_label = [curr_method, ', perturbation=', num2str(perturbation_SFPL)];
        fprintf('%s\n', curr_label);
        [solution, obj_values] = SFPL(W0, Xtrain, ytrain, Xtest, ytest, num_iters_SFPL, batch_size, perturbation_SFPL, IS_NON_SMOOTH, loss_handle, subgrad_handle, lmo_handle, gap_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, stoptime);
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
            semilogy(obj_values(:, 3), cumsum(obj_values(:, 4)) ./ (1 : size(obj_values, 1))', 'DisplayName', curr_label); hold on;
            xlabel('time');
            ylabel('average cost');
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
    training_accuracy = accuracy_fn(solution, Xtrain, ytrain);
    fprintf('training loss: %f, accuracy: %f\n', training_loss, training_accuracy);
end
drawnow;

%% Definitions of loss function, gradient, and linear optimization oracle
if IS_NON_SMOOTH
    output_file_name = [DATA_ROOT, 'results_', dataset, '_SVM_l1_auto_save.mat'];
else
    output_file_name = [DATA_ROOT, 'results_', dataset, '_SVM_smooth_auto_save.mat'];
end
save(output_file_name, 'selected_methods', 'obj_values_cell');
beep;

grid on;

function loss = loss_fn(W, X, y, alpha, lambda)
    % smoothed hinge loss plus l_1 regularizer
    z = squeeze(sum(W .* X, [1, 2])) .* y;
    loss = 0;
    tmpz = z(z <= 0);
    if ~isempty(tmpz)
        loss = loss + sum(alpha / (alpha + 1.0) - tmpz);
    end
    tmpz = z(z > 0 & z < 1);
    if ~isempty(tmpz)
        loss = loss + sum(tmpz.^(alpha + 1) ./ (alpha + 1) + alpha / (alpha + 1) - tmpz);
    end
    loss = loss / length(y) + lambda * sum(sum(abs(W)));
end

function accuracy = accuracy_fn(W, X, y)
    scores = sign(sum(W .* X, [1, 2]));
    scores(scores == 0) = 1;
    accuracy = sum(squeeze(scores) == y) / length(y);
end

function grad = grad_fn(W, X, y, alpha)
    z = (reshape(X, [numel(W), size(X, 3)])' * reshape(W, [numel(W), 1])) .* y;
    tmpz_idx = (z <= 0);
    tmp_coef = zeros(size(z));
    length_tmpz_idx = sum(tmpz_idx);
    if length_tmpz_idx > 0
        tmp_coef(tmpz_idx) = -1 * y(tmpz_idx);
    end
    tmpz_idx = (z > 0 & z < 1);
    length_tmpz_idx = sum(tmpz_idx);
    if length_tmpz_idx > 0
        tmp_coef(tmpz_idx) = (z(tmpz_idx).^alpha - 1) .* y(tmpz_idx);
    end
    grad = reshape(reshape(X, [numel(W), size(X, 3)]) * (tmp_coef ./ length(tmp_coef)), size(W));
end

function grad_diff = grad_diff_fn(W, W_old, rho, X, y, alpha)
    % compute grad_fn(W, X, y, alpha) - (1 - rho) * grad_fn(W_old, X, y, alpha)
    z = (reshape(X, [numel(W), size(X, 3)])' * reshape(W, [numel(W), 1])) .* y;
    z_old = (reshape(X, [numel(W_old), size(X, 3)])' * reshape(W_old, [numel(W_old), 1])) .* y;

    tmp_coef = zeros(size(z));
    tmpz_idx = (z <= 0);
    length_tmpz_idx = sum(tmpz_idx);
    if length_tmpz_idx > 0
        tmp_coef(tmpz_idx) = -1 * y(tmpz_idx);
    end
    tmpz_idx = (z > 0 & z < 1);
    length_tmpz_idx = sum(tmpz_idx);
    if length_tmpz_idx > 0
        tmp_coef(tmpz_idx) = (z(tmpz_idx).^alpha - 1) .* y(tmpz_idx);
    end

    tmp_coef_old = zeros(size(z_old));
    tmpz_old_idx = (z_old <= 0);
    length_tmpz_old_idx = sum(tmpz_old_idx);
    if length_tmpz_old_idx > 0
        tmp_coef_old(tmpz_old_idx) = -1 * y(tmpz_old_idx);
    end
    tmpz_old_idx = (z_old > 0 & z_old < 1);
    length_tmpz_old_idx = sum(tmpz_old_idx);
    if length_tmpz_old_idx > 0
        tmp_coef_old(tmpz_old_idx) = (z_old(tmpz_old_idx).^alpha - 1) .* y(tmpz_old_idx);
    end

    grad_diff = reshape(reshape(X, [numel(W), size(X, 3)]) * ((tmp_coef - (1 - rho) .* tmp_coef_old) ./ length(tmp_coef)), size(W));
end


function subgrad = subgrad_fn(W, X, y, alpha, lambda)
    subgrad = grad_fn(W, X, y, alpha) + lambda .* sign(W);
end

function res = lmo_trace_norm_fn(A, radius)  % lmo for nuclear norm constraint || X ||_* <= radius
    [lvec, ~, rvec] = rsvd(A, 1);
    res = lvec*(-radius)*rvec';
end

function fw_gap = gap_fn(W, X, y)
    % @NOTE no need to compute the Frank-Wolfe gap, simply set to zero
    fw_gap = 0;
end

function ret = grad_moreau_smoothed_reg_fn(W, beta, lambda)
    ret = (W - sign(W) .* max(abs(W) - beta * lambda, 0)) ./ beta;
end

function ret = get_F_norm_ball_radius(model_radius)
    ret = model_radius;
end
