function [solution, obj_values] = PFOCO_SG(W1, sample_fn, num_iters, sub_batch_size, eta_coef, eta_exp, reg_coef, loss_fn, grad_fn, lmo_fn, gap_fn, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, beta0, grad_smoothed_reg_fn)

    if ~IS_ADVERSARIAL
        error('Go home, MSTORM_FW only applies to the adversarial setting.');
    end

    if exist('beta0', 'var') && beta0 > 0
        IS_NON_SMOOTH = true;
    else
        IS_NON_SMOOTH = false;
    end

    %% initialization
    overhead_time = 0.0;

    grad_estimate = zeros(size(W1));  % gradient estimator

    obj_values = zeros(fix(num_iters / print_freq) + 1, 5);
    if IS_CALCULATING_REGRET
        obj_values(1, :) = [0, 0, 0.0, 0.0, 0.0];
    else
        obj_values(1, :) = [0, 0, 0.0, 0, 0];
    end

    if IS_NON_SMOOTH
        num_oracles = num_iters;
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
        W = W1;
        grad_estimate(:) = 0.0;  % gradient estimator
        for k = 1 : num_oracles
            eta = min(eta_coef * 2 / (k + 1)^eta_exp, 1.0);
            %% FPL prediction
            V = lmo_fn(grad_cell_arr{k} .* reg_coef + noise_cell_arr{k});
            % update grad_estimate
            curr_samples = sample_fn(sub_batch_size, t);
            grad_estimate = grad_fn(W, curr_samples);
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

            running_time = t_current - overhead_time;
            if IS_CALCULATING_REGRET && IS_ADVERSARIAL
                curr_loss = loss_fn(W, t) - loss_fn(optimal_solution, t);
                curr_gap = gap_fn(W);
                obj_values(fix(t / print_freq) + 1, :) = [t, t * sub_batch_size, running_time, curr_loss, curr_gap];
            elseif IS_ADVERSARIAL
                curr_loss = loss_fn(W, t);
                curr_gap = 0.0;
                obj_values(fix(t / print_freq) + 1, :) = [t, t * sub_batch_size, running_time, curr_loss, curr_gap];
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
