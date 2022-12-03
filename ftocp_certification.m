% ftocp_certification
%
% Technical University of Munich
% University of Toronto Institute for Aerospace Studies
% Learning Systems and Robotics Lab
%
% Author
% Lukas Brunke: lukas.brunke@tum.de

function [x_sol, u_sol] = ftocp_certification(f, n, m, p, N, L_x, L_u, l, P_f, tighten_by, alpha, x_init, u_unsafe)
    % Safety filter optimal control problem

    opti = casadi.Opti();

    % optimiization variable
    x = opti.variable(n, N);
    u = opti.variable(m, N - 1);

    u_diff = u(:, 1) - u_unsafe;

    % objective function
    opti.minimize(u_diff' * u_diff);

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