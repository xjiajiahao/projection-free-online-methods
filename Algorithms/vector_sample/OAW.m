function [solution, obj_values] = OAW(W1, sample_fn, num_iters, batch_size, eta_coef, eta_exp, loss_fn, grad_fn, lmo_fn, gap_fn, print_freq, IS_CALCULATING_REGRET, optimal_value, stoptime, IS_SPARSE)
    if nargin < 15
        IS_SPARSE = false;
    end

    % initialization
    % @NOTE W1 has to be 0
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

    active_cell_arr = {};
    weight_cell_arr = {};
    num_FW_steps_taken = 0;

    t_start = tic;  % timing

    for t = 1 : num_iters
        % sample an index
        W_old = W;
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
        % compute the away direction
        if ~isempty(active_cell_arr)
            max_dot_prod_val = -inf;
            max_dot_prod_idx = 0;
            for tmp_i = 1 : length(active_cell_arr)
                curr_dot_prod = sum(sum(active_cell_arr{tmp_i} .* grad_estimate));
                if curr_dot_prod > max_dot_prod_val
                    max_dot_prod_val = curr_dot_prod;
                    max_dot_prod_idx = tmp_i;
                end
            end
        end
        % determine whether we take a FW step or away step or drop step
        if isempty(active_cell_arr) || sum(sum(V .* grad_estimate)) + max_dot_prod_val <= 2 * sum(sum(W .* grad_estimate))
            % FW step
            num_FW_steps_taken = num_FW_steps_taken + 1;
            eta = min(2 * eta_coef / (num_FW_steps_taken + 1)^eta_exp, 1.0);
            W = (1 - eta) * W + eta * V;
            % fw_op(active_cell_arr, weight_cell_arr, eta, V);
            IS_V_ACTIVE = false;
            idx_V = 0;
            for tmp_i = 1 : length(active_cell_arr)
                if isequal(V, active_cell_arr{tmp_i})
                    IS_V_ACTIVE = true;
                    idx_V = tmp_i;
                    break;
                end
            end
            for tmp_i = 1 : length(weight_cell_arr)
                weight_cell_arr{tmp_i} = weight_cell_arr{tmp_i} * (1 - eta);
            end
            if IS_V_ACTIVE
                weight_cell_arr{idx_V} = weight_cell_arr{idx_V} + eta;
            else  % insert new vertex
                active_cell_arr{end+1} = V;
                weight_cell_arr{end+1} = eta;
            end
        else
            % away or drop step
            eta_max = 1.0/(1 - weight_cell_arr{max_dot_prod_idx}) - 1;
            if eta_max >= eta  % away step
                eta = eta_coef / (num_FW_steps_taken + 1)^eta_exp;
                W = (1 + eta) * W - eta * active_cell_arr{max_dot_prod_idx};
                % away_op(active_cell_arr, weight_cell_arr, eta, max_dot_prod_idx);
                for tmp_i = 1 : length(weight_cell_arr)
                    weight_cell_arr{tmp_i} = weight_cell_arr{tmp_i} * (1 + eta);
                end
                weight_cell_arr{max_dot_prod_idx} = weight_cell_arr{max_dot_prod_idx} - eta;
            else  % drop step
                eta = eta_max;
                W = (1 + eta) * W - eta * active_cell_arr{max_dot_prod_idx};
                % drop_op(active_cell_arr, weight_cell_arr, eta, max_dot_prod_idx);
                weight_cell_arr(max_dot_prod_idx) = [];
                active_cell_arr(max_dot_prod_idx) = [];
                for tmp_i = 1 : length(weight_cell_arr)
                    weight_cell_arr{tmp_i} = weight_cell_arr{tmp_i} * (1 + eta);
                end
            end
        end

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
