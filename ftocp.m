% ftocp
%
% Technical University of Munich
% University of Toronto Institute for Aerospace Studies
% Learning Systems and Robotics Lab
%
% Author
% Lukas Brunke: lukas.brunke@tum.de

function [x_sol, u_sol] = ftocp(f, n, m, p, N, Q, R, L_x, L_u, l, P_f, tighten_by, alpha, x_init)
    % Solves a finite-time optimal control problem
    opti = casadi.Opti();

    % optimiization variable
    x = opti.variable(n, N);
    u = opti.variable(m, N - 1);

    % stage cost
    stage_cost = 0;
    for i = 1 : N - 1
        stage_cost = stage_cost + x(:, i)' * Q * x(:, i) + u(:, i)' * R * u(:, i);
    end

    % terminal cost
    terminal_cost = x(:, N)' * P_f * x(:, N);

    % objective function
    opti.minimize(stage_cost + terminal_cost);

    % initial state constraint
    opti.subject_to(x(:, 1) == x_init);

    for i = 1 : N - 1
        % dynamic constraints
        opti.subject_to(x(:, i + 1) ==  f(x(:, i), u(:, i)));        
        
        % state and input constraints
        for j = 1 : p
            opti.subject_to(L_x(j, :) * x(:, i) + L_u(j, :) * u(:, i) <= l(j) - tighten_by(i));
        end
    end

    % terminal constraint
    opti.subject_to(x(:, N)' * P_f * x(:, N) <= alpha^2);

    % solve optimization
    opti.solver('ipopt');
    sol = opti.solve();

    % retrieve solution
    x_sol = sol.value(x);
    u_sol = sol.value(u);
end
