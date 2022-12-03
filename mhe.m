% mhe
%
% Technical University of Munich
% University of Toronto Institute for Aerospace Studies
% Learning Systems and Robotics Lab
%
% Author
% Lukas Brunke: lukas.brunke@tum.de

function [w_hat_sol, x_hat_sol, y_hat_sol, cost] = mhe(f, M_t, n, n_y, q, prev_outputs, prev_inputs, e_max, w_max, eta, C, E, F, P_o, eps_factor, sigma_1_factor, sigma_2_factor, prev_estimates)
    % Moving horizon estimation
    opti = casadi.Opti();
        
    if size(prev_estimates, 2) <= M_t
        u = prev_inputs;
        prev_estimate = prev_estimates(:, 1);
    else
        u = prev_inputs(:, end - M_t + 1: end);
        prev_estimate = prev_estimates(:, end - M_t + 1);
    end
    
    y = prev_outputs(:, end - M_t + 1: end);
    
    % optimiization variable
    w_hat = opti.variable(q, M_t);
    x_hat = opti.variable(n, M_t);
    y_hat = opti.variable(n_y, M_t);
    
    % stage cost
    stage_cost = 0;
    for i = 1 : M_t 
        delta_y = y(:, i) - y_hat(:, i);
        sigma_1_cost = eta^(M_t - i) * (eps_factor * (sigma_1_factor * (w_max + sqrt(w_hat(:, i)' * w_hat(:, i)))^2));
        sigma_2_cost = eta^(M_t - i) * (eps_factor * (sigma_2_factor * (delta_y' * delta_y)));
        stage_cost = stage_cost + sigma_1_cost + sigma_2_cost;
    end

    % arrival cost
    arrival_cost = 0;
    arrival_diff = x_hat(:, 1) - prev_estimate;
    arrival_cost = eta^(M_t) *( arrival_diff' * P_o * arrival_diff + 2 * sqrt(e_max) * sqrt(arrival_diff' * P_o * arrival_diff) ); 

    cost = stage_cost + arrival_cost;
    
    % objective function
    opti.minimize(cost);

    for i = 1 : M_t
        if i < M_t
            opti.subject_to(x_hat(:, i + 1) == f(x_hat(:, i), u(:, i)) + E * w_hat(:, i));        
        end
        opti.subject_to(y_hat(:, i) == C * x_hat(:, i) + F * w_hat(:, i));
    end
    
    % construct nominal solution
    x_hat_nominal = zeros(n, M_t);
    x_hat_nominal(:, 1) = prev_estimate;
    for i = 1 : M_t - 1
        x_hat_nominal(:, i + 1) = full(f(x_hat_nominal(:, i), u(:, i)));
    end
    
    % set initial values for optimization problem
    opti.set_initial(x_hat, x_hat_nominal + 1e-3 * ones(n, M_t));
    opti.set_initial(y_hat, y);
    opti.set_initial(w_hat, 1e-1 * w_max * ones(q, M_t));
    
    % solve optimization
    p_opts = struct('expand', true);
    s_opts = struct('max_iter', 1e3); %% iteration limitation
    opti.solver('ipopt', p_opts, s_opts);
    try
        sol = opti.solve();
        % retrieve solution
        w_hat_sol = sol.value(w_hat)
        x_hat_sol = sol.value(x_hat)
        y_hat_sol = sol.value(y_hat)
        cost = sol.value(Cost)
    catch
        disp("retrieve failed solution")
        w_hat_sol = opti.debug.value(w_hat)
        x_hat_sol = opti.debug.value(x_hat)
        y_hat_sol = opti.debug.value(y_hat)
        cost = opti.debug.value(cost)
    end
    
end