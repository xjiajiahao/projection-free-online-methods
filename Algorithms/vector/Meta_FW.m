function [solution, obj_values] = Meta_FW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, sub_batch_size, eta_coef, eta_exp, rho_coef, rho_exp, reg_coef, loss_fn, grad_fn, lmo_fn, gap_fn, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, beta0, grad_smoothed_reg_fn, IS_SPARSE)
    if nargin < 24
        IS_SPARSE = false;
    end


    feature_dims = repmat({':'}, 1, ndims(Xtrain) - 1);
    data_size = length(ytrain);

    if ~IS_ADVERSARIAL
        error('Go home, MFW only applies to the adversarial setting.');
    end

    if exist('beta0', 'var') && beta0 > 0
        IS_NON_SMOOTH = true;
    else
        IS_NON_SMOOTH = false;
    end

    %% initialization
    overhead_time = 0.0;

    if IS_SPARSE
        grad_estimate = sparse(size(W1, 1), size(W1, 2));  % gradient estimator
    else
        grad_estimate = zeros(size(W1));  % gradient estimator
    end

    obj_values = zeros(fix(num_iters / print_freq) + 1, 5);
    if IS_CALCULATING_REGRET
        obj_values(1, :) = [0, 0, 0.0, 0.0, 0.0];
    else
        obj_values(1, :) = [0, 0, 0.0, loss_fn(W1, Xtrain, ytrain), gap_fn(W1, Xtrain, ytrain)];
    end

    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    % obj_values(1, :) = [0, 0, 0.0, 0.0, 0.0];

    if sub_batch_size < batch_size
        num_oracles = ceil(num_iters^(3/2));
    else
        num_oracles = ceil(sqrt(num_iters));
    end

    grad_cell_arr = cell(num_oracles, 1);
    noise_cell_arr = cell(num_oracles, 1);

    % reg_coef = 1/sqrt(num_iters);
    reg_coef = reg_coef / sqrt(num_iters);

    t_start = tic;  % timing

    for k = 1 : num_oracles
        grad_cell_arr{k} = grad_estimate;  % zeros
        noise_cell_arr{k} = -0.5 + rand(size(W1));
    end

    %% main loop
    for t = 1 : num_iters
        global_indices = mod((t-1) * batch_size, data_size) + 1: mod((t-1) * batch_size, data_size) + batch_size;

        W = W1;
        if IS_SPARSE
            grad_estimate = sparse(size(W1, 1), size(W1, 2));  % gradient estimator
        else
            grad_estimate = zeros(size(W1));  % gradient estimator
        end

        for k = 1 : num_oracles
            eta = min(eta_coef / (k + 3)^eta_exp, 1.0);
            rho = min(rho_coef * 2 / (k + 3)^rho_exp, 1.0);

            %% FPL prediction
            V = lmo_fn(grad_cell_arr{k} .* reg_coef + noise_cell_arr{k});
            % update grad_estimate (pretending that we already have f_t)
            if sub_batch_size < batch_size
                idx = randi([1, batch_size], [sub_batch_size, 1]);
                Xtmp_tmp = Xtrain(feature_dims{:}, global_indices(idx));
                ytmp_tmp = ytrain(global_indices(idx));
                stoch_grad = grad_fn(W, Xtmp_tmp, ytmp_tmp);
            else
                idx = (1 : batch_size)';
                stoch_grad = grad_fn(W, Xtrain(feature_dims{:}, global_indices), ytrain(global_indices));
            end
            if size(stoch_grad, 2) < size(grad_estimate, 2)
                grad_estimate = (1 - rho) * (grad_estimate);
                grad_estimate(:, ytmp(idx)) = grad_estimate(:, ytmp(idx)) + rho * stoch_grad;
            else
                grad_estimate = (1 - rho) * (grad_estimate) + rho * stoch_grad;
            end
            % feedback the linear objective function to FPL
            if IS_NON_SMOOTH
                beta = beta0 / sqrt(k + 1);
                grad_cell_arr{k} = grad_cell_arr{k} + grad_estimate + grad_smoothed_reg_fn(W, beta);
            else
                grad_cell_arr{k} = grad_cell_arr{k} + grad_estimate;
            end
            % update W
            W = (1 - eta) * W + eta * V;
        end

        %% evaluate loss function value and FW gap
        if mod(t, print_freq) == 0
            t_current = toc(t_start);
            Xtmp = Xtrain(feature_dims{:}, global_indices);
            ytmp = ytrain(global_indices);

            running_time = t_current - overhead_time;
            if IS_CALCULATING_REGRET && IS_ADVERSARIAL
                curr_loss = loss_fn(W, Xtmp, ytmp) - loss_fn(optimal_solution, Xtmp, ytmp);
                curr_gap = gap_fn(W, Xtmp, ytmp);
                obj_values(fix(t / print_freq) + 1, :) = [t, t * batch_size, running_time, curr_loss, curr_gap];
            elseif IS_ADVERSARIAL
                curr_loss = loss_fn(W, Xtmp, ytmp);
                curr_gap = 0.0;
                obj_values(fix(t / print_freq) + 1, :) = [t, t * batch_size, running_time, curr_loss, curr_gap];
            end
            overhead_time = overhead_time + toc(t_start) - t_current;
        end
    end

    W = W1;
    for k = 1 : num_oracles
        eta = min(eta_coef / (k + 1)^eta_exp, 1.0);
        %% FPL prediction
        V = lmo_fn(grad_cell_arr{k} .* reg_coef + noise_cell_arr{k});
        W = (1 - eta) * W + eta * V;
    end
    solution = W;
end
