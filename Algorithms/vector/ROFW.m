function [solution, obj_values] = ROFW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, eta_coef, eta_exp, reg_coef, loss_fn, grad_fn, lmo_fn, gap_fn, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, stoptime)
    % initialization
    feature_dims = repmat({':'}, 1, ndims(Xtrain) - 1);
    overhead_time = 0.0;

    data_size = size(Xtrain, ndims(Xtrain));
    W = W1;  % copy on write
    observed_grad_sum = zeros(size(W1));  % gradient estimator

    obj_values = zeros(fix(num_iters / print_freq) + 1, 5);
    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    if IS_CALCULATING_REGRET
        obj_values(1, :) = [0, 0, 0.0, 0.0, 0.0];
    else
        obj_values(1, :) = [0, 0, 0.0, loss_fn(W, Xtrain, ytrain), gap_fn(W, Xtrain, ytrain)];
    end

    if IS_CALCULATING_REGRET
        optimal_value = loss_fn(optimal_solution, Xtrain, ytrain);
    end

    t_start = tic;  % timing

    for t = 1 : num_iters
        eta = min(eta_coef / t^eta_exp, 1.0);
        reg_param = reg_coef / t^(3/4);
        % sample an index
        if IS_ADVERSARIAL
            idx = mod((t-1) * batch_size, data_size) + 1: mod((t-1) * batch_size, data_size) + batch_size;
        else
            idx = randi([1, data_size], [batch_size, 1]);
        end
        W_old = W;
        % update grad_estimate
        Xtmp = Xtrain(feature_dims{:}, idx);
        ytmp = ytrain(idx);
        stoch_grad = grad_fn(W, Xtmp, ytmp);
        if size(stoch_grad, 2) < size(observed_grad_sum, 2)
            observed_grad_sum(:, ytmp) = observed_grad_sum(:, ytmp) + stoch_grad;
        else
            observed_grad_sum = observed_grad_sum + stoch_grad;
        end
        grad_estimate = (observed_grad_sum) + (W - W1) * 2 / reg_param;
        % LMO
        V = lmo_fn(grad_estimate);
        % update W
        W = (1 - eta) * W + eta * V;

        % evaluate loss function value and FW gap
        if mod(t, print_freq) == 0
            t_current = toc(t_start);
            running_time = t_current - overhead_time;
            if IS_CALCULATING_REGRET && IS_ADVERSARIAL
                curr_loss = loss_fn(W_old, Xtmp, ytmp) - loss_fn(optimal_solution, Xtmp, ytmp);
                curr_gap = gap_fn(W_old, Xtmp, ytmp);
                obj_values(fix(t / print_freq) + 1, :) = [t, t * batch_size, running_time, curr_loss, curr_gap];
            elseif IS_CALCULATING_REGRET
                curr_loss = loss_fn(W_old, Xtrain, ytrain) - optimal_value;
                curr_gap = gap_fn(W, Xtmp, ytmp);
                obj_values(fix(t / print_freq) + 1, :) = [t, t * batch_size, running_time, curr_loss, curr_gap];
            elseif IS_ADVERSARIAL
                curr_loss = loss_fn(W_old, Xtmp, ytmp);
                curr_gap = 0.0;
                obj_values(fix(t / print_freq) + 1, :) = [t, t * batch_size, running_time, curr_loss, curr_gap];
            else
                curr_loss = loss_fn(W, Xtrain, ytrain);
                curr_gap = 0.0;
                obj_values(fix(t / print_freq) + 1, :) = [t, t * batch_size, running_time, curr_loss, curr_gap];
            end
            overhead_time = overhead_time + toc(t_start) - t_current;

            if running_time >= stoptime
                obj_values(fix(t / print_freq) + 2 : end, :) = [];
                break;
            end
        end
    end
    solution = W;
end
