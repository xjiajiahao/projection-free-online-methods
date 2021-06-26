function [solution, obj_values] = FW(W1, num_iters, batch_size, eta_coef, eta_exp, beta0, loss_fn, grad_fn, lmo_fn, gap_fn, grad_smoothed_reg_fn, print_freq, IS_CALCULATING_REGRET, stoptime)

    if beta0 == 0
        IS_NON_SMOOTH = false;
    else
        IS_NON_SMOOTH = true;
    end

    % initialization
    overhead_time = 0.0;

    W = W1;  % copy on write

    obj_values = zeros(fix(num_iters / print_freq) + 1, 5);
    % each row records [#iters, #SFO, running_time, loss_value, FW_gap];
    if IS_CALCULATING_REGRET
        obj_values(1, :) = [0, 0, 0.0, 0.0, 0.0];
    else
        obj_values(1, :) = [0, 0, 0.0, loss_fn(W), gap_fn(W)];
    end


    t_start = tic;  % timing
    % rng(1);

    for t = 1 : num_iters
        eta = min(eta_coef / (t + 1)^eta_exp, 1.0);
        % sample an index
        % update grad_estimate
        grad_estimate = grad_fn(W);
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
                curr_loss = loss_fn(W_old);
                curr_gap = gap_fn(W);
            else
                curr_loss = loss_fn(W);
                curr_gap = gap_fn(W);
            end
            obj_values(fix(t / print_freq) + 1, :) = [t, t * batch_size * 2, running_time, curr_loss, curr_gap];
            overhead_time = overhead_time + toc(t_start) - t_current;

            if running_time >= stoptime
                obj_values(fix(t / print_freq) + 2 : end, :) = [];
                break;
            end
        end
    end
    solution = W;
end
