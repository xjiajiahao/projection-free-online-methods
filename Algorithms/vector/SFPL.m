function [solution, obj_values] = SFPL(W1, Xtrain, ytrain, Xtest, ytest, num_iters, batch_size, perturbation, IS_NON_SMOOTH, loss_fn, grad_fn, lmo_fn, gap_fn, print_freq, IS_ADVERSARIAL, IS_CALCULATING_REGRET, optimal_solution, stoptime)
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

    t_start = tic;  % timing

    if IS_NON_SMOOTH
        block_size = num_iters;
    else
        block_size = ceil(nthroot(num_iters, 2));
    end

    delta = perturbation * sqrt(numel(W) * num_iters );

    for t = 1 : num_iters
        W_old = W;

        W(:) = 0;  % W = 0
        for i = 1 : block_size
            % sample xi from the unit ball
            xi = randn(size(W));  % generate a sample from the standard normal distribution
            xi = xi ./ sqrt(sum(sum(xi.^2)));  % projection to the unit sphere
            tmp_radius = nthroot(rand(), numel(W));
            xi = xi .* (tmp_radius * delta);
            V = lmo_fn(observed_grad_sum + xi);
            W = W + V;
        end
        W = W ./ block_size;

        % observe f_t, evalute the gradient of f_t
        % sample an index
        if IS_ADVERSARIAL
            idx = mod((t-1) * batch_size, data_size) + 1: mod((t-1) * batch_size, data_size) + batch_size;
        else
            idx = randi([1, data_size], [batch_size, 1]);
        end
        Xtmp = Xtrain(feature_dims{:}, idx);
        ytmp = ytrain(idx);
        stoch_grad = grad_fn(W, Xtmp, ytmp);
        observed_grad_sum = observed_grad_sum + stoch_grad;

        % evaluate loss function value and FW gap
        if mod(t, print_freq) == 0
            t_current = toc(t_start);
            running_time = t_current - overhead_time;
            if IS_CALCULATING_REGRET && IS_ADVERSARIAL
                curr_loss = loss_fn(W_old, Xtmp, ytmp) - loss_fn(optimal_solution, Xtmp, ytmp);
                curr_gap = gap_fn(W_old, Xtmp, ytmp);
                obj_values(fix(t / print_freq) + 1, :) = [t, (t+1)*t/2 * batch_size, running_time, curr_loss, curr_gap];
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
