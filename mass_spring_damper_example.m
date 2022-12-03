% mass_spring_damper_example: Main script for 'Robust Predictive
% Output-Feedback Safety Filter for Uncertain Nonlinear Control Systems'
%
% Technical University of Munich
% University of Toronto Institute for Aerospace Studies
% Learning Systems and Robotics Lab
%
% Author
% Lukas Brunke: lukas.brunke@tum.de

%%
clear all
close all
clc

%% Problem set up

use_online_estimation = true;
run_uncertified = false;
run_mpc = false;
run_filter = true;

assert(run_uncertified + run_mpc + run_filter == 1);

% dimensions
n_x = 2;
m = 1;
n_y = 1;
q_x = 2;
q_y = 1;
q = q_x + q_y;

% initial state
x_0 = [0.79; 0.7];

% Mass spring damper system parameters
M = 1;
k_0 = 0.33;
h_d = 1.1;

% time parameters
delta_t = 0.25;
N = 40; % control horizon
N_est = 10; % estimation horizon

% cost parameters
Q = eye(n_x);
R = 0.01;

% constraints
X_limits = [1, - 0.85, 0.85;
            2, - 2,    2];
U_limits = [1,  -6,    6];
W_limits = [1, - 0.01,  0.01;
            2, - 0.01,  0.01];
V_limits = [1, - 0.01, 0.01];
w_max = 0.01;

% set up polytopic representation of box constraints
[L_x, l_x] = box2polytopic(X_limits, n_x);
[L_u, l_u] = box2polytopic(U_limits, m);
[L_w, l_w] = box2polytopic(W_limits, q_x);
[L_v, l_v] = box2polytopic(V_limits, q_y);
L_wv = blkdiag(L_w, L_v);
l_wv = [l_w; l_v];

% num constraints
p_x = length(l_x);
p_u = length(l_u);
p = p_x + p_u;
L_x = [L_x; zeros(p - p_x, n_x)];
L_u = [zeros(p - p_u, m); L_u];
l = [l_x; l_u]; 

%% Dynamics using Casadi
import casadi.*

% Casadi variables
x = MX.sym('x', n_x);  
u = MX.sym('u', m);
y = MX.sym('y', n_y);
w = MX.sym('w', q_x);
v = MX.sym('v', q_y);

% Casadi continuous-time dynamics for nonlinear mass spring damper system
dynamics = [x(2);
            1 / M * (- k_0 * x(1) * exp(- x(1)) - h_d * x(2) + u(1))];
f_cont = Function('x_dot', {x, u}, {dynamics}, {'x', 'u'}, {'x_dot'});
A_jac = Function('A_jac', {x, u}, {jacobian(f_cont(x, u), x)}, {'x', 'u'}, {'A_jac'});
B_jac = Function('B_jac', {x, u}, {jacobian(f_cont(x, u), u)}, {'x', 'u'}, {'B_jac'});

% Casadi discrete-time system using Runge-Kutta 4
phi_1 = Function('phi_1', {x, u}, {dynamics}, {'x', 'u'}, {'phi_1'});
phi_2 = Function('phi_2', {x, u}, {f_cont(x + 0.5 * delta_t * phi_1(x, u), u)}, {'x', 'u'}, {'phi_2'});
phi_3 = Function('phi_3', {x, u}, {f_cont(x + 0.5 * delta_t * phi_2(x, u), u)}, {'x', 'u'}, {'phi_3'});
phi_4 = Function('phi_4', {x, u}, {f_cont(x + delta_t * phi_3(x, u), u)}, {'x', 'u'}, {'phi_4'});
rungeKutta = x + delta_t / 6 * (phi_1(x, u) + 2 * phi_2(x, u) + 2 * phi_3(x, u) + phi_4(x, u));
rungeKutta_func = Function('rk4', {x, u}, {rungeKutta}, {'x', 'u'}, {'rk4'});

% Linearize discrete-time system (RK4) ...
A_rk4 = Function('A_r', {x, u}, {jacobian(rungeKutta_func(x, u), x)},...
             {'x','u'},...
             {'A_jacobian'});
B_rk4 = Function('B_r', {x, u}, {jacobian(rungeKutta_func(x, u), u)},...
             {'x','u'},...
             {'B_jacobian'});

% output dynamics 
C = [1, 0];

% Disturbance matrices
E = diag([delta_t, delta_t / M]);
F = 1;

%% Find feasible state and input combinations

% number of points we consider per dimension
points_per_dim = 10; 

% set up grid
X1 = linspace(X_limits(1, 2), X_limits(1, 3), points_per_dim);
X2 = linspace(X_limits(2, 2), X_limits(2, 3), points_per_dim);
U = linspace(U_limits(2), U_limits(3), points_per_dim);

% initialize sets
feasible_combinations = []; % set that contains all feasible states and inputs
tmp_reachable_set = [];
reachable_sets = {}; % cell array that contains the reachable set for all feasible states and inputs

num_feasible = 0;
num_reachable = 0;

% determine feasible states and inputs
for id_1 = 1 : points_per_dim
    for id_2 = 1 : points_per_dim % 2
        x = [X1(id_1); X2(id_2)];  % loop through entire grid
        for id_u1 = 1 : points_per_dim
            u = U(id_u1);  % loop through all inputs
            % Note: x and u are inside constraints by design

            % determine next state
            x_next = full(rungeKutta_func(x, u));
            if in_limits(x_next, X_limits) && in_limits(u, U_limits)
                for id_u2 = 1 : points_per_dim
                    u_next = U(id_u2);  % loop through all inputs
                    % determine next next state
                    x_next2 = full(rungeKutta_func(x_next, u_next));
                    if in_limits(x_next2, X_limits) && in_limits(u_next, U_limits)
                        tmp_reachable_set = [tmp_reachable_set, [x_next; u_next]];
                    end
                end
            end

            % only add states if the reachable set is non-empty
            if size(tmp_reachable_set, 1) > 0
                num_feasible = num_feasible + 1;
                num_reachable = num_reachable + size(tmp_reachable_set, 2);
                feasible_combinations = [feasible_combinations, [x; u]];
                reachable_sets{num_feasible} = tmp_reachable_set;
            end
            tmp_reachable_set = [];

        end
    end
end
num_feasible
num_reachable

%% Design incremental Lyapunov function V_s

% Design parameters for incremental Lyapunov function synthesis
rho_s = 0.78;
cs_j = 0.49;

X_s = sdpvar(n_x, n_x);
Y_s = sdpvar(m, n_x, 'full');

Cost = - logdet(X_s);

Constraints = [];
            
for feasible_id = 1 : num_feasible
    % get state and input from feasible state and input combinations
    x = feasible_combinations(1:n_x, feasible_id);
    u = feasible_combinations(n_x + 1 : n_x + m, feasible_id);
    
    % determine the Jacobians
    A_r = full(A_rk4(x, u));
    B_r = full(B_rk4(x, u));
    
    AXBY = A_r * X_s + B_r * Y_s;
    
    Constraints = [Constraints;
                   [X_s, AXBY;
                    AXBY', rho_s^2 * X_s] >= 0];
end
            
for j = 1 : p
    LXLY = L_x(j, :) * X_s + L_u(j, :) * Y_s;
    Constraints = [Constraints;
                   [cs_j^2, LXLY;
                    LXLY', X_s] >= 0];                 
end
            
options = sdpsettings('solver', 'mosek', 'verbose', 1);

sol = optimize(Constraints, Cost, options);

X_s = value(X_s);
Y_s = value(Y_s);
P_s = inv(X_s);
K_s = Y_s * P_s; 

%% Check validity of incremental Lyapunov function V_s

c_su = sqrt(max(eig(P_s)));
c_sl = sqrt(min(eig(P_s)));

C_sj = zeros(p, 1);

for j = 1 : p
    C_sj(j) = norm(((L_x(j, :) + L_u(j, :) * K_s)) * inv(sqrtm((P_s))));
end

c_sj = max(C_sj);

Rho_s = zeros(num_feasible, 1);

num_valid_controllers = 0;
for feasible_id = 1 : num_feasible
    % get state and input from feasible state and input combinations
    x = feasible_combinations(1:n_x, feasible_id);
    u = feasible_combinations(n_x + 1 : n_x + m, feasible_id);
    
    % determine the Jacobians
    A_r = full(A_rk4(x, u));
    B_r = full(B_rk4(x, u));
    
    is_valid = 0;
    ABK = A_r + B_r * K_s;
    eigs = eig(ABK);
    
    % check decay rate
    rho_s_test_squared = sdpvar(1);
    Constraints = [ABK' * P_s * ABK <= rho_s_test_squared * P_s];
    Cost = rho_s_test_squared;
    
    options = sdpsettings('solver', 'mosek', 'verbose', 0);
    sol = optimize(Constraints, Cost, options);
    
    Rho_s(feasible_id) = sqrt(value(rho_s_test_squared));
    
    for i = 1 : n_x 
        if norm(eigs(i)) <= 1
            is_valid = is_valid + 1;
        end
    end
    
    if is_valid == n_x
        num_valid_controllers = num_valid_controllers + 1;
    end
end

rho_s = max(Rho_s);
if rho_s >= 1
    disp("Decay rate is invalid!")
else
    disp("Decay rate is valid!")
end

%% Design incremental Lyapunov function V_o

% Design parameters for incremental Lyapunov function synthesis
rho_o = 0.63;
L_max = 0.67;
w_o1 = 0.016;
w_o2 = 0.04;
c_oj = 0.49;

% Optimization variables
P_o = sdpvar(n_x, n_x);
Z_o = sdpvar(n_x, n_y);

% Objective
Cost = - logdet(P_o);

Constraints = [];

ZC = Z_o * C;

% Constraint on the deviation between observer dynamics and nominal
% prediction
Constraints = [P_o, ZC;
               ZC', L_max^2 * P_o] >= 0;

% Constraints for the decay rate for all feasible states
for feasible_id = 1 : num_feasible
    % get state and input from feasible state and input combinations
    x = feasible_combinations(1:n_x, feasible_id);
    u = feasible_combinations(n_x + 1 : n_x + m, feasible_id);
    
    % determine the Jacobians
    A_r = full(A_rk4(x, u));

    PAZC = P_o * A_r - ZC;

    Constraints = [Constraints;
                   [P_o, PAZC;
                    PAZC', rho_o^2 * P_o] >= 0];
end

% Constraints such that the state estimation error is bounded for each
% state constraint 
for j = 1 : p
    Constraints = [Constraints;
                   [P_o, L_x(j, :)';
                    L_x(j, :), c_oj^2] >= 0];                 
end

% Constraints on the impact of the process and measurement noise
for v_id = 1 : 2
    v = V_limits(1 + v_id);
    ZFv = Z_o * F * v;
    Constraints = [Constraints;
                   [P_o, ZFv;
                    ZFv', w_o1^2] >= 0];
    for w1_id = 1 : 2
        for w2_id = 1 : 2
            w = [W_limits(1, 1 + w1_id); W_limits(2, 1 + w2_id)];
            PEw = P_o * E * w;
            Constraints = [Constraints;
                           [P_o, ZFv - PEw;
                            ZFv' - PEw', w_o2^2] >= 0];
        end
    end
end

options = sdpsettings('solver', 'sdpt3', 'verbose', 1, 'sdpt3.maxit', 100);

sol = optimize(Constraints, Cost, options);

% reconstruct solution
Z_o = value(Z_o);
P_o = value(P_o);
L =  inv(P_o) * Z_o; 

%%
c_ou = sqrt(max(eig(P_o)));
c_ol = sqrt(min(eig(P_o)));

C_oj = zeros(p, 1);

for j = 1 : p
    C_oj(j) = norm(L_x(j, :) * inv(sqrtm((P_o))));
end

c_oj = max(C_oj);

L_max = norm(L * C * inv(sqrtm(P_o)));

w_o1 = norm(L * F * V_limits(3));
w_o2 = norm(sqrtm(P_o) * (L * F * V_limits(3) - E * [W_limits(1, 2); W_limits(2, 3)]));

Rho_o = zeros(num_feasible, 1);

num_valid_observers = 0;
for feasible_id = 1 : num_feasible
    % get state and input from feasible state and input combinations
    x = feasible_combinations(1:n_x, feasible_id);
    u = feasible_combinations(n_x + 1 : n_x + m, feasible_id);
    
    % determine the Jacobians
    A_r = full(A_rk4(x, u));
    
    is_valid = 0;
    ALC = A_r - L * C;
    eigs = eig(ALC);
    
    % check decay rate
    rho_o_test_squared = sdpvar(1);
    Constraints = [ALC' * P_o * ALC <= rho_o_test_squared * P_o];
    Cost = rho_o_test_squared;
    
    options = sdpsettings('solver', 'mosek', 'verbose', 0);
    sol = optimize(Constraints, Cost, options);
    
    Rho_o(feasible_id) = sqrt(value(rho_o_test_squared));
    
    for i = 1 : n_x 
        if norm(eigs(i)) <= 1
            is_valid = is_valid + 1;
        end
    end
    
    if is_valid == n_x
        num_valid_observers = num_valid_observers + 1;
    end
end

rho_o = max(Rho_o);
if rho_o >= 1
    disp("Decay rate is invalid!")
else
    disp("Decay rate is valid!")
end

%% Terminal ingredients

% design parameter for terminal Lyapunov function synthesis
rho_f = 0.9;

A_lin = full(A_rk4(zeros(n_x, 1), zeros(m, 1)));
B_lin = full(B_rk4(zeros(n_x, 1), zeros(m, 1)));
[K_f, ~] = dlqr(A_lin, B_lin, Q, R);
K_f = - K_f;

A_cl = A_lin + B_lin * K_f;

mu = 1.2;

P_f = sdpvar(n_x, n_x);

Constraints = [P_f >= 0;
               A_cl' * P_f * A_cl - P_f == - mu * (Q + K_f' * R * K_f)];

options = sdpsettings('solver', 'mosek', 'verbose', 1);
sol = optimize(Constraints, [], options);

P_f = value(P_f);
rho_f = sqrt(1 - min(eig(sqrtm(inv(P_f)) * (Q + K_f' * R * K_f) * sqrtm(inv(P_f)))));

eps_1 = c_su / c_sl * sqrt(max(eig(P_f))) * w_o1;
eps_2 = eps_1 * L_max / w_o1;

%% Constraint tightening constants

% Size of the terminal set
alpha_bar = 5;

% linear parameter for constraint tightening 
alpha_1 = 0.085;

a = zeros(N, 1);
b = zeros(N, 1);

a(1) = c_oj;

for i = 1 : N - 1
    a(i + 1) = rho_o * a(i) + c_sj * rho_s^(i - 1) * c_su * L_max; 
    b(i + 1) = b(i) + a(i) * w_o2 + c_sj * rho_s^(i - 1) * c_su * w_o1;
end

LLKPf = (L_x + L_u * K_f) * inv(sqrtm(P_f));

normed_LLKPf = zeros(p, 1);

alpha_0_candidates = zeros(p, 1);
alpha_1_candidates = zeros(p, 1);

for i = 1 : p
    alpha_0_candidates(i) = (1 - b(N)) / norm(LLKPf(i, :));
    alpha_1_candidates(i) = a(N) / norm(LLKPf(i, :));
end

alpha_0 = min(alpha_0_candidates);

alpha_1_upper = 1 / w_o2 * ((1 - rho_f) * alpha_0 - rho_s^N * eps_1);
alpha_1_lower = rho_s^N * eps_2 / (rho_f - rho_o);

% Check value for alpha_1
assert(alpha_1 <= alpha_1_upper && alpha_1 >= max([alpha_1_candidates; alpha_1_lower]))

%% Check Terminal ingredients

phi = linspace(0, 2 * pi, 50);
x_sphere = alpha_bar * cos(phi);
y_sphere = alpha_bar * sin(phi);

coords_ellipse = inv(sqrtm(P_f)) * [x_sphere; y_sphere];

% w_none = zeros(q, 1);
num_random_vectors = 10000;

% Sample points from gamma^2 * unit sphere
delta_x = randsphere(num_random_vectors, n_x, alpha_bar)';

% Transform sampled points into ellipsoid to span the candidate terminal
% set and shift around reference point x_r
dx_transform = inv(sqrtm(P_f)) * delta_x;

% initialize counter
num_valid = 0;
inside_set = 0;

for i = 1 : num_random_vectors
    % get sampled vector
    x_i = dx_transform(:, i);
    
    % get terminal control input
    u = K_f * x_i;
    
    % simulate system using control input
    x_plus = full(rungeKutta_func(x_i, u));
    
    % evaluate stage cost and terminal costs
    stage = x_i' * Q * x_i + u' * R * u;
    V_f = x_i' * P_f * x_i;
    V_f_plus = x_plus' * P_f * x_plus;
    
    % check Lyapunov condition for terminal cost
    if V_f_plus <= V_f - stage
        num_valid = num_valid + 1;
    end
end

%% Online estimation bounds using detectability
epsilon = 0.1; 

eta = (1 + epsilon) * rho_o;

% kappa functions for the definition of the identical incremental Lyapunov
% function
sigma_1 = @(r) 2 * (1 + epsilon) / epsilon * (sqrt(max(eig(E' * P_o * E))) + sqrt(max(eig(F' * L' * P_o * L * F))))^2 * r^2;
sigma_2 = @(r) 2 * (1 + epsilon) / epsilon * max(eig(L' * P_o * L)) * r^2;

eps_factor = 2 * (1 + epsilon) / epsilon;
sigma_1_factor = (sqrt(max(eig(E' * P_o * E))) + sqrt(max(eig(F' * L' * P_o * L * F))))^2;
sigma_2_factor = max(eig(L' * P_o * L));

%% Preparation for MHE

e_max = w_o2 / (1 - rho_o);
E_mhe = [E, zeros(q_x, q_y)];
F_mhe = [zeros(q_y, q_x), F];

%% Set up robust nonlinear MPC
close all

V_bar_o = w_o2 / (1 - rho_o);
e_bar_inits = [V_bar_o; 0];

used_online_bound = 0;
used_mhe = 0;

num_steps = 60;

x_init = x_0;
x_hat_init = x_init;
y_t = C * x_init;

e_bar_o = 0;

prev_inputs = [];
prev_estimates = [x_hat_init];
prev_outputs = [y_t];
prev_e_bar = [e_bar_o];

X_cl = zeros(n_x, num_steps + 1);
X_cl(:, 1) = x_hat_init;

% Set up ellipse plotting
num_ellipse_points = 50;
alpha = linspace(0, 2* pi, num_ellipse_points);
r = 1;

% Create ellipses for the tubes
x_ellipse_s = inv(sqrtm(P_s)) * r * [cos(alpha); sin(alpha)];
x_ellipse_o = inv(sqrtm(P_o)) * r * [cos(alpha); sin(alpha)];

% find maximum extension of tube for prediction error in x_1 direction for
% plotting
[x1_max_extension, x1_index] = max(x_ellipse_s(1, :));

% Create ellipse for the terminal set
x_ellipse_f = inv(sqrtm(P_f)) * r * [cos(alpha); sin(alpha)];

figure 
hold on

plot(X_limits(1, 3) * ones(2, 1), [X_limits(2, 2), X_limits(2, 3)], 'k-', 'LineWidth', 3)
plot(X_limits(1, 2) * ones(2, 1), [X_limits(2, 2), X_limits(2, 3)], 'k-', 'LineWidth', 3)
plot([X_limits(1, 2), X_limits(1, 3)], X_limits(2, 3) * ones(2, 1), 'k-', 'LineWidth', 3)
plot([X_limits(1, 2), X_limits(1, 3)], X_limits(2, 2) * ones(2, 1), 'k-', 'LineWidth', 3)

% Select at which time steps the uncertainty sets are plotted
selected_times = [1, 2, 5, N];

% period for sinusoidal uncertified control input
period = 20;

for i = 1 : num_steps    
    % unsafe control input 
    u_desired = sin(pi / period * i);

    % Precompute constraint tightening
    e_k = zeros(N + 1, 1);
    s_k = zeros(N + 1, 1);
    
    % propagate uncertainty tubes
    for j = 1 : N + 1
        e_k(j, 1) = (1 - rho_o^(j - 1)) / (1 - rho_o) * w_o2 + rho_o^(j - 1) * e_bar_o;
        if j > 1
            s_k(j, 1) = rho_s * s_k(j - 1, 1) + c_su * (L_max * e_k(j - 1, 1) + w_o1);
        end
    end

    % precomputed tightening
    l_k = c_sj * s_k + c_oj * e_k;

    % determine tightening of terminal constraint set
    alpha_i = alpha_0 - alpha_1 * e_bar_o;
    
    if run_uncertified
        % ------ apply unsafe control input ------ 
        u = u_desired;
    end
    
    if run_mpc
        % ------ robust output-feedback MPC ------
        % solve finite-time optimal control problem
        [x_sol, u_sol] = ftocp(rungeKutta_func, n_x, m, p, N, Q, R, L_x, L_u, l, P_f, l_k, alpha_i, x_hat_init);
    end
    
    if run_filter
        % ------ safety filter -------
        [x_sol, u_sol] = ftocp_certification(rungeKutta_func, n_x, m, p, N, L_x, L_u, l, P_f, l_k, alpha_i, x_hat_init, u_desired);
        u = u_sol(:, 1);
    end

    if i == 1
        plot(x_sol(1, :), x_sol(2, :), 'r--', 'DisplayName', 'open-loop traj')
        plot(x_sol(1, selected_times), x_sol(2, selected_times), 'b.', 'MarkerSize', 10)

        s_k = 0;
        e_k = e_bar_o;

        % plot open-loop tube
        for k = 1 : selected_times(end)

            if any(selected_times == k)
                ellipse = fill(x_sol(1, k) + s_k * x_ellipse_s(1, :), x_sol(2, k) + s_k * x_ellipse_s(2, :), 'c', 'FaceAlpha', 0.2);
                ellipse = fill(x_sol(1, k) + s_k * x_ellipse_s(1, x1_index) + e_k * x_ellipse_o(1, :), x_sol(2, k) + s_k * x_ellipse_s(2, x1_index) + e_k * x_ellipse_o(2, :), 'g', 'FaceAlpha', 0.2);

                plot(x_sol(1, k) + s_k * x_ellipse_s(1, x1_index), x_sol(2, k) + s_k * x_ellipse_s(2, x1_index), 'g.', 'MarkerSize', 10)

                % remove tube sets from legend
                ellipse.Annotation.LegendInformation.IconDisplayStyle = 'off';
            end
            
            s_k = rho_s * s_k + c_su * (L_max * e_k + w_o1);  % propagate prediction error tube size
            e_k = rho_o * e_k + w_o2;  % propagate estimation error tube size
        end
        ellipse = fill(alpha_i * x_ellipse_f(1, :), alpha_i * x_ellipse_f(2, :), 'm', 'FaceAlpha', 0.2);
        
        % remove tube sets from legend
        ellipse.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end

    % propagate estimate
    x_hat_init = full(rungeKutta_func(x_hat_init, u)) + L * (y_t - C * x_hat_init);

    % get disturbance
    w = (W_limits(:, 3) - W_limits(:, 2)) .* rand(q_x, 1) - W_limits(:, 3);
    v = (V_limits(3) - V_limits(2)) .* rand(q_y, 1) - V_limits(3);

    % propagate system
    x_init = full(rungeKutta_func(x_init, u)) + E * w;
    y_t = C * x_init + F * v;
    
    % update list of previously applied inputs and measurements
    prev_inputs = [prev_inputs, u];
    prev_outputs = [prev_outputs, y_t];
    
    if use_online_estimation
        % online estimation error bound
        y_hat = C * x_hat_init;
        w_hat_k = norm(E' * inv(E * E') * L * (y_hat - y_t));
        e_bar_online = eta * e_bar_o + sigma_1(w_max + w_hat_k) + sigma_2(norm(y_hat - y_t));
        e_bar_offline = rho_o * e_bar_o + w_o2;  % propagate estimation error tube size
        e_bar_o = min(e_bar_offline, e_bar_online);
        if e_bar_online <= e_bar_offline
            disp("-----------Using online bound-----------")
            used_online_bound = used_online_bound + 1;
        else
            disp("-----------Using offline bound-----------")
        end
    else
        % offline estimation error bound
        e_bar_o = rho_o * e_bar_o + w_o2;
    end
    
    M_t = min(i + 1, N_est);
    [w_hat_sol, x_hat_sol, y_hat_sol, mhe_cost] = mhe(rungeKutta_func, M_t, n_x, n_y, q, prev_outputs, prev_inputs, e_max, w_max, eta, C, E_mhe, F_mhe, P_o, eps_factor, sigma_1_factor, sigma_2_factor, prev_estimates);
    e_bar_mhe = mhe_cost + eta^M_t * prev_e_bar(end - M_t + 2);

    if e_bar_mhe < e_bar_o
        disp("-----------Using MHE bound-----------")
        x_hat_init = x_hat_sol(:, end);
        e_bar_o = e_bar_mhe;
        used_mhe = used_mhe + 1;
    else
        disp("-----------Using Luenberger bound-----------")
    end
    
    % update list of estimates and errors
    prev_estimates = [prev_estimates, x_hat_init];
    prev_e_bar = [prev_e_bar, e_bar_o];

    X_cl(:, i + 1) = x_hat_init;  % save closed-loop solution
end

plot(X_cl(1, :), X_cl(2, :), 'k-')

xlabel('x_1')
ylabel('x_2')
