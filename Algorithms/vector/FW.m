function [solution, obj_values] = FW(W1, Xtrain, ytrain, Xtest, ytest, num_iters, eta_coef, eta_exp, loss_fn, grad_fn, lmo_fn, gap_fn, beta0, grad_smoothed_reg_fn)

    if beta0 == 0
        IS_NON_SMOOTH = false;
    else
        IS_NON_SMOOTH = true;
    end

    % initialization
    overhead_time = 0.0;

    [~, data_size] = size(Xtrain);
    W = W1;  % copy on write

    obj_values = zeros(num_iters + 1, 5);
    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    obj_values(1, :) = [0, 0, 0.0, loss_fn(W, Xtrain, ytrain), gap_fn(W, Xtrain, ytrain)];

    t_start = tic;  % timing

    grad_estimate = grad_fn(W, Xtrain, ytrain);

    for t = 1 : num_iters
        eta = min(eta_coef / (t + 1)^eta_exp, 1.0);
        % LMO
        if IS_NON_SMOOTH
            beta = beta0 / sqrt(t + 1);
            grad_smoothed_reg = grad_smoothed_reg_fn(W, beta);
            V = lmo_fn(grad_estimate + grad_smoothed_reg);
        else
            V = lmo_fn(grad_estimate);
        end
        W = ( 1 - eta) * W + eta * V;
        % update grad_estimate
        grad_estimate = grad_fn(W, Xtrain, ytrain);

        % evaluate loss function value and FW gap
        t_current = toc(t_start);
        running_time = t_current - overhead_time;
        curr_loss = loss_fn(W, Xtrain, ytrain);
        curr_gap = gap_fn(W, Xtrain, ytrain);
        obj_values(t + 1, :) = [t, t * data_size, running_time, curr_loss, curr_gap];
        overhead_time = overhead_time + toc(t_start) - t_current;
    end
    solution = W;
end
