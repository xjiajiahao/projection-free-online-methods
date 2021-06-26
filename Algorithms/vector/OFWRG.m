function [solution, obj_values] = OFWRG(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, sub_batch_size, eta_coef, eta_exp, rho_coef, rho_exp, loss_fn, grad_fn, lmo_fn, gap_fn, print_freq, IS_CALCULATING_REGRET, optimal_value, beta0, grad_smoothed_reg_fn, stoptime)

    feature_dims = repmat({':'}, 1, ndims(Xtrain) - 1);

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

    data_size = size(Xtrain, ndims(Xtrain));
    W = W1;  % copy on write

    obj_values = zeros(fix(num_iters / print_freq) + 1, 5);
    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    if IS_CALCULATING_REGRET
        obj_values(1, :) = [0, 0, 0.0, 0.0, 0.0];
    else
        obj_values(1, :) = [0, 0, 0.0, loss_fn(W, Xtrain, ytrain), gap_fn(W, Xtrain, ytrain)];
    end

    t_start = tic;  % timing

    for t = 1 : num_iters
        eta = min(eta_coef / t^eta_exp, 1.0);
        rho = min(rho_coef / t^rho_exp, 1.0);
        % sample an index
        idx = randi([1, data_size], [batch_size, 1]);
        if sub_batch_size < batch_size
            sub_idx = randi([1, batch_size], [sub_batch_size, 1]);
            idx = idx(sub_idx);
        end
        Xtmp = Xtrain(feature_dims{:}, idx);
        ytmp = ytrain(idx);
        % update grad_estimate
        if t == 1
            grad_estimate = grad_fn(W, Xtmp, ytmp);
        else
            stoch_grad = grad_fn(W, Xtmp, ytmp);
            stoch_grad_old = grad_fn(W_old, Xtmp, ytmp);
            grad_estimate = (1 - rho) * (grad_estimate - stoch_grad_old) + stoch_grad;
        end
        % LMO
        if IS_NON_SMOOTH
            beta = beta0 / sqrt(t + 1);
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
            obj_values(fix(t / print_freq) + 1, :) = [t, t * sub_batch_size * 2, running_time, curr_loss, curr_gap];
            overhead_time = overhead_time + toc(t_start) - t_current;

            if running_time >= stoptime
                obj_values(fix(t / print_freq) + 2 : end, :) = [];
                break;
            end
        end
    end
    solution = W;
end
