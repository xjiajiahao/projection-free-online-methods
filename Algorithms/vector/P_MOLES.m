function [solution, obj_values] = P_MOLES(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, sub_batch_size, eta_coef, eta_exp, lambda_coef, gamma_coef, radius, loss_fn, subgrad_fn, lmo_fn, gap_fn, print_freq, IS_CALCULATING_REGRET, optimal_value, stoptime)
    if sub_batch_size > batch_size
        sub_batch_size = batch_size;
    end

    % initialization
    overhead_time = 0.0;

    [~, data_size] = size(Xtrain);
    W = W1;  % copy on write
    W_tilde = W1;
    Z = W1;
    Z_tilde = W1;
    lambda = lambda_coef / sqrt(num_iters);
    if eta_exp == 0
        eta_coef = eta_coef / num_iters^(3/2);
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
        eta = eta_coef / t^eta_exp;
        beta = 8.0 / (t * lambda) + 1.0 / (t * eta);
        gamma = 2.0 * gamma_coef / (t+1);
        Y = (1 - gamma) * W + gamma * Z;
        Y_tilde = (1 - gamma) * W_tilde + gamma * Z_tilde;
        % LMO
        Z = lmo_fn(Y - Y_tilde);
        % sample an index
        idx = randi([1, data_size], [batch_size, 1]);
        if sub_batch_size < batch_size
            sub_idx = randi([1, batch_size], [sub_batch_size, 1]);
            idx = idx(sub_idx);
        end
        Xtmp = Xtrain(:, idx);
        ytmp = ytrain(idx);
        stoch_subgrad = subgrad_fn(W, Xtmp, ytmp);
        Z_tilde = Z_tilde - ((Y_tilde - Y) ./ lambda + stoch_subgrad) ./ beta;
        norm_Z_tilde = norm(Z_tilde, 'fro');
        if radius / norm_Z_tilde < 1
            Z_tilde = Z_tilde .* (radius / norm_Z_tilde);
        end
        % update W
        W_old = W;
        W = (1 - gamma) * W + gamma * Z;
        W_tilde = (1 - gamma) * W_tilde + gamma * Z_tilde;

        % evaluate loss function value and FW gap
        if mod(t, print_freq) == 0
            t_current = toc(t_start);
            running_time = t_current - overhead_time;
            if IS_CALCULATING_REGRET
                curr_loss = loss_fn(W_old, Xtrain, ytrain) - optimal_value;
                curr_gap = gap_fn(W_old, Xtrain, ytrain);
                obj_values(fix(t / print_freq) + 1, :) = [t, (t+1)*t/2 * sub_batch_size, running_time, curr_loss, curr_gap];
            else
                curr_loss = loss_fn(W, Xtrain, ytrain);
                curr_gap = gap_fn(W, Xtrain, ytrain);
                obj_values(fix(t / print_freq) + 1, :) = [t, t * sub_batch_size, running_time, curr_loss, curr_gap];
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
