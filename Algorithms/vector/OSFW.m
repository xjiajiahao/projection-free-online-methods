function [solution, obj_values] = OSFW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, sub_batch_size, eta_coef, eta_exp, rho_coef, rho_exp, loss_fn, grad_fn, lmo_fn, gap_fn, print_freq, IS_CALCULATING_REGRET, optimal_value, beta0, grad_smoothed_reg_fn, stoptime, IS_SPARSE)
    if nargin < 23
        IS_SPARSE = false;
    end

    if exist('beta0', 'var') && beta0 > 0
        IS_NON_SMOOTH = true;
    else
        IS_NON_SMOOTH = false;
    end

    if sub_batch_size > batch_size
        sub_batch_size = batch_size;
    end


    % initialization
    overhead_time = 0.0;

    [~, data_size] = size(Xtrain);
    W = W1;  % copy on write

    if IS_SPARSE
        grad_estimate = sparse(size(W1, 1), size(W1, 2));  % gradient estimator
    else
        grad_estimate = zeros(size(W1));  % gradient estimator
    end

    obj_values = zeros(fix(num_iters / print_freq) + 1, 5);
    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    if IS_CALCULATING_REGRET
        obj_values(1, :) = [0, 0, 0.0, 0.0, 0.0];
    else
        obj_values(1, :) = [0, 0, 0.0, loss_fn(W, Xtrain, ytrain), gap_fn(W, Xtrain, ytrain)];
    end

    t_start = tic;  % timing

    for t = 1 : num_iters
        eta = min(eta_coef * 9 / (t + 8)^eta_exp, 1.0);
        rho = min(rho_coef * 4 / (t + 7)^rho_exp, 1.0);
        % sample an index
        idx = randi([1, data_size], [batch_size, 1]);
        if sub_batch_size < batch_size
            sub_idx = randi([1, batch_size], [sub_batch_size, 1]);
            idx = idx(sub_idx);
        end

        Xtmp = Xtrain(:, idx);
        ytmp = ytrain(idx);
        % update grad_estimate
        stoch_grad = grad_fn(W, Xtmp, ytmp);
        grad_estimate = (1 - rho) * (grad_estimate) + rho * stoch_grad;
        % LMO
        if IS_NON_SMOOTH
            beta = beta0 / sqrt(t + 8);
            V = lmo_fn(grad_estimate + grad_smoothed_reg_fn(W, beta));
        else
            V = lmo_fn(grad_estimate);
        end
        % update W
        W_old = W;
        W = (1 - eta) * W + eta * V;

        % evaluate loss function value and FW gap
        if mod(t, print_freq) == 0
            t_current = toc(t_start);
            running_time = t_current - overhead_time;
            if IS_CALCULATING_REGRET
                curr_loss = loss_fn(W_old, Xtrain, ytrain) - optimal_value;
                curr_gap = gap_fn(W, Xtmp, ytmp);
            else
                curr_loss = loss_fn(W, Xtrain, ytrain);
                curr_gap = gap_fn(W, Xtrain, ytrain);
            end
            obj_values(fix(t / print_freq) + 1, :) = [t, t * sub_batch_size, running_time, curr_loss, curr_gap];
            overhead_time = overhead_time + toc(t_start) - t_current;

            if running_time >= stoptime
                obj_values(fix(t / print_freq) + 2 : end, :) = [];
                break;
            end
        end
    end
    solution = W;
end
