function [solution, obj_values] = OFW(W1, sample_fn, num_iters, batch_size, eta_coef, eta_exp, loss_fn, grad_fn, lmo_fn, gap_fn, print_freq, IS_CALCULATING_REGRET, optimal_value, stoptime, IS_SPARSE)
    if nargin < 15
        IS_SPARSE = false;
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

    observed_samples_cell_arr = cell(num_iters, 1);

    t_start = tic;  % timing

    for t = 1 : num_iters
        eta = min(2 * eta_coef / (t + 1)^eta_exp, 1.0);
        % sample an index and store data
        observed_samples_cell_arr{t} = sample_fn(batch_size);

        % update grad_estimate
        if IS_SPARSE
            grad_estimate = sparse(size(W1, 1), size(W1, 2));  % gradient estimator
        else
            grad_estimate = zeros(size(W1));
        end

        for tmp_t = 1 : t
            grad_estimate = grad_estimate + grad_fn(W, observed_samples_cell_arr{tmp_t});
        end
        grad_estimate = grad_estimate ./ t;
        % LMO
        V = lmo_fn(grad_estimate);
        % update W
        W_old = W;
        W = (1 - eta) * W + eta * V;

        % evaluate loss function value and FW gap
        if mod(t, print_freq) == 0
            t_current = toc(t_start);
            running_time = t_current - overhead_time;
            if IS_CALCULATING_REGRET
                curr_loss = loss_fn(W_old) - optimal_value;
                curr_gap = gap_fn(W);
            else
                curr_loss = loss_fn(W);
                curr_gap = gap_fn(W);
            end
            obj_values(fix(t / print_freq) + 1, :) = [t, (t+1)*t/2 * batch_size, running_time, curr_loss, curr_gap];
            overhead_time = overhead_time + toc(t_start) - t_current;

            if running_time >= stoptime
                obj_values(fix(t / print_freq) + 2 : end, :) = [];
                break;
            end
        end
    end
    solution = W;
end
