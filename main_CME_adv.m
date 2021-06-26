% clear;
%% Load data
addpath('./Algorithms/vector_sample/');
addpath('./Config/CME/');
DATA_ROOT = './Data/';

% Set parameters (feel free to modify parameters in this section)
if ~exist('IS_TUNING_PARAMETERS', 'var') || IS_TUNING_PARAMETERS == false
    toy_adv_64;
end

optimal_value_file_name = [DATA_ROOT, dataset, '_CME_opt.mat'];
optimal_solution = [];
if isfile(optimal_value_file_name)
    load(optimal_value_file_name, 'optimal_value', 'optimal_solution');
    IS_OPTIMAL_VALUE_AVAILABLE = true;
else
    IS_OPTIMAL_VALUE_AVAILABLE = false;
end

if ~IS_OPTIMAL_VALUE_AVAILABLE && IS_CALCULATING_REGRET
    IS_CALCULATING_REGRET = false;
end

%% Run algorithms
num_iters_Meta_FW = num_iters_base;
num_iters_Meta_OFWRG = num_iters_base;
num_iters_PFOCO_SG = num_iters_base;

W0 = zeros(dim);

loss_handle = @(X, t) loss_fn(X, sqrtSU_collections, t);
grad_handle = @grad_fn;
grad_full_handle = @(X, t) grad_full_fn(X, sqrtSU_collections, t);
grad_diff_handle = @grad_diff_fn;
subgrad_handle = @(X, samples) subgrad_fn(X, samples);
sample_handle = @(num_samples, t) generate_sample(num_samples, dim, sqrtSU_collections, t);
lmo_handle = @(V) lmo_trace_norm_fn(V, model_radius);
grad_moreau_smoothed_reg_handle = @grad_moreau_smoothed_reg_fn;  % not used
gap_handle = @(X) gap_fn(X, model_radius);
ball_radius = get_F_norm_ball_radius(model_radius);


obj_values_cell = cell(length(selected_methods), 1);
solutions_cell = cell(length(selected_methods), 1);

for method_idx = 1 : length(selected_methods)
    curr_method = selected_methods{method_idx};
    if strcmp(curr_method, 'Meta_OFWRG') == true
        if IS_NON_SMOOTH
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_Meta_OFWRG), ', \eta exp=', num2str(eta_exp_Meta_OFWRG), ', \rho coef=', num2str(rho_coef_Meta_OFWRG), ', \rho exp=', num2str(rho_exp_Meta_OFWRG), ', reg coef=', num2str(reg_coef_Meta_OFWRG), ', \beta=', num2str(beta0_Meta_OFWRG), ', batch=', num2str(sub_batch_size)];
        else
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_Meta_OFWRG), ', \eta exp=', num2str(eta_exp_Meta_OFWRG), ', \rho coef=', num2str(rho_coef_Meta_OFWRG), ', \rho exp=', num2str(rho_exp_Meta_OFWRG), ', reg coef=', num2str(reg_coef_Meta_OFWRG), ', batch=', num2str(sub_batch_size)];
        end
        fprintf('%s\n', curr_label);
        [solution, obj_values] = Meta_OFWRG(W0, sample_handle, num_iters_Meta_OFWRG, sub_batch_size, eta_coef_Meta_OFWRG, eta_exp_Meta_OFWRG, rho_coef_Meta_OFWRG, rho_exp_Meta_OFWRG, reg_coef_Meta_OFWRG, loss_handle, grad_handle, lmo_handle, gap_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, beta0_Meta_OFWRG, grad_moreau_smoothed_reg_handle, grad_diff_handle);
    elseif strcmp(curr_method, 'Meta_FW') == true
        if IS_NON_SMOOTH
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_Meta_FW), ', \eta exp=', num2str(eta_exp_Meta_FW), ', \rho coef=', num2str(rho_coef_Meta_FW), ', \rho exp=', num2str(rho_exp_Meta_FW), ', reg coef=', num2str(reg_coef_Meta_FW), ', \beta=', num2str(beta0_Meta_FW), ', batch=', num2str(sub_batch_size)];
        else
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_Meta_FW), ', \eta exp=', num2str(eta_exp_Meta_FW), ', \rho coef=', num2str(rho_coef_Meta_FW), ', \rho exp=', num2str(rho_exp_Meta_FW), ', reg coef=', num2str(reg_coef_Meta_FW), ', batch=', num2str(sub_batch_size)];
        end
        fprintf('%s\n', curr_label);
        [solution, obj_values] = Meta_FW(W0, sample_handle, num_iters_Meta_FW, sub_batch_size, eta_coef_Meta_FW, eta_exp_Meta_FW, rho_coef_Meta_FW, rho_exp_Meta_FW, reg_coef_Meta_FW, loss_handle, grad_handle, lmo_handle, gap_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, beta0_Meta_FW, grad_moreau_smoothed_reg_handle);
    elseif strcmp(curr_method, 'PFOCO_SG') == true
        if IS_NON_SMOOTH
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_PFOCO_SG), ', \eta exp=', num2str(eta_exp_PFOCO_SG), ', reg coef=', num2str(reg_coef_PFOCO_SG), ', \beta=', num2str(beta0_PFOCO_SG), ', batch=', num2str(sub_batch_size)];
        else
            curr_label = [curr_method, ', \eta coef=', num2str(eta_coef_PFOCO_SG), ', \eta exp=', num2str(eta_exp_PFOCO_SG), ', reg coef=', num2str(reg_coef_PFOCO_SG), ', batch=', num2str(sub_batch_size)];
        end
        fprintf('%s\n', curr_label);
        [solution, obj_values] = PFOCO_SG(W0, sample_handle, num_iters_PFOCO_SG, sub_batch_size, eta_coef_PFOCO_SG, eta_exp_PFOCO_SG, reg_coef_PFOCO_SG, loss_handle, grad_handle, lmo_handle, gap_handle, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, beta0_PFOCO_SG, grad_moreau_smoothed_reg_handle);
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
            semilogy(obj_values(:, 3), (obj_values(:, 4) - optimal_value) ./ (obj_values(1, 4) - optimal_value), 'DisplayName', curr_label); hold on;
            xlabel('time (s)');
            ylabel('loss value');
            legend('show');
        end
    else
        semilogy(obj_values(:, 3), obj_values(:, 4), 'DisplayName', curr_label); hold on;
        xlabel('#iterations');
        ylabel('loss value');
        legend('show');
    end
    drawnow;
    obj_values_cell{method_idx} = obj_values;
    solutions_cell{method_idx} = solution;

end

%% Definitions of loss function, gradient, and linear optimization oracle
output_file_name = [DATA_ROOT, 'results_', dataset, '_CME_auto_save.mat'];
save(output_file_name, 'selected_methods', 'obj_values_cell', 'solutions_cell');
beep;

grid on;

% objective function: covariance matrix estiamtion
% f(X) = E_w ||X - w w^T||_F^2, s.t., X \in S_+^d, ||X||_* <= radius
function loss = loss_fn(X, sqrtSU_collections, t)
    t_clip = mod(t - 1, length(sqrtSU_collections)) + 1;
    model_covariance = sqrtSU_collections{t_clip}' * sqrtSU_collections{t_clip};
    loss = norm(X - model_covariance,'fro')^2 / 2;
end

function ret = generate_sample(num_samples, dim, sqrtSU_collections, t)
    t_clip = mod(t - 1, length(sqrtSU_collections)) + 1;
    ret = randn(num_samples, dim) * sqrtSU_collections{t_clip};
end

function grad = grad_fn(X, samples)
    num_samples = size(samples, 1);
    grad = X - ((samples ./ num_samples)' * samples);
end

function grad_full = grad_full_fn(X, sqrtSU_collections, t)
    t_clip = mod(t - 1, length(sqrtSU_collections)) + 1;
    model_covariance = sqrtSU_collections{t_clip}' * sqrtSU_collections{t_clip};
    grad_full = X - model_covariance;
end


function grad_diff = grad_diff_fn(X, samples, X_old, rho)
    num_samples = size(samples, 1);
    grad_diff = X - (1-rho) * X_old - (samples .* (rho / num_samples))' * samples;
end

function subgrad = subgrad_fn(X, samples)
    num_samples = size(samples, 1);
    subgrad = X - ((samples ./ num_samples)' * samples);
end

function ret = lmo_trace_norm_fn(A, radius)  % lmo for nuclear norm constraint || X ||_* <= radius
    A = (A + A') ./ 2;
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
