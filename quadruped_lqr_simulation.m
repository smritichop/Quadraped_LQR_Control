%% ============================================================
% 2D SAGITTAL PLANE QUADRUPED LQR CONTROL SIMULATION
% Project: Simulink model of quadruped using LQR control to improve trajectory following
% 
% This simulation demonstrates LQR control advantages over PD control
% for a simplified 2D quadruped robot in the sagittal plane.
%
% Reference: MIT Cheetah 3 - "Dynamic Locomotion Through Convex MPC"
%% ============================================================

clear; clc; close all;

%% ============================================================
% 1) PHYSICAL PARAMETERS (Based on MIT Cheetah 3)
%% ============================================================

% Robot parameters
params.m = 43;           % mass [kg]
params.Iyy = 2.1;        % pitch moment of inertia [kg*m^2]
params.g = 9.81;         % gravity [m/s^2]

% Geometry (sagittal plane)
params.L = 0.3;          % half body length (hip to CoM) [m]
params.h_nom = 0.35;     % nominal standing height [m]

% Leg parameters
params.l_leg = 0.4;      % leg length [m]

% Friction coefficient (for force constraints)
params.mu = 0.6;

% Force limits per leg
params.f_min = 10;       % minimum normal force [N]
params.f_max = 500;      % maximum normal force [N]

fprintf('=== 2D Sagittal Plane Quadruped LQR Simulation ===\n');
fprintf('Robot mass: %.1f kg\n', params.m);
fprintf('Pitch inertia: %.2f kg*m^2\n', params.Iyy);
fprintf('Nominal height: %.2f m\n\n', params.h_nom);

%% ============================================================
% 2) STATE-SPACE MODEL (2D Sagittal Plane)
%% ============================================================
% State vector: x = [px; pz; theta; vx; vz; omega]
%   px    - horizontal position [m]
%   pz    - vertical position (height of CoM) [m]  
%   theta - pitch angle [rad] (positive = nose up)
%   vx    - horizontal velocity [m/s]
%   vz    - vertical velocity [m/s]
%   omega - pitch rate [rad/s]
%
% Control input: u = [f1x; f1z; f2x; f2z]
%   f1x, f1z - front leg ground reaction force [N]
%   f2x, f2z - rear leg ground reaction force [N]
%
% Equations of motion (Newton-Euler):
%   m * ax = f1x + f2x
%   m * az = f1z + f2z - m*g
%   Iyy * alpha = r1 x f1 + r2 x f2
%
% where r1, r2 are vectors from CoM to foot contact points

nx = 6;  % number of states
nu = 4;  % number of control inputs (both legs in stance)

% Continuous-time A matrix (dynamics)
% State: [px; pz; theta; vx; vz; omega]
A_c = [0 0 0 1 0 0;    % dx/dt = vx
       0 0 0 0 1 0;    % dz/dt = vz
       0 0 0 0 0 1;    % dtheta/dt = omega
       0 0 0 0 0 0;    % dvx/dt = forces/m
       0 0 0 0 0 0;    % dvz/dt = forces/m - g
       0 0 0 0 0 0];   % domega/dt = torques/Iyy

% Continuous-time B matrix
% For stance with both legs on ground:
% Front foot at (+L, 0) relative to CoM in body frame
% Rear foot at (-L, 0) relative to CoM in body frame
% Torque = r x F = L*fz (front) and -L*fz (rear) for small angles

B_c = [0           0           0           0;
       0           0           0           0;
       0           0           0           0;
       1/params.m  0           1/params.m  0;
       0           1/params.m  0           1/params.m;
       params.L/params.Iyy  0  -params.L/params.Iyy  0];

% Gravity vector (constant disturbance)
g_vec = [0; 0; 0; 0; -params.g; 0];

% Discretize the system
dt = 0.001;  % 1 kHz control loop (matches MIT Cheetah)
A_d = eye(nx) + A_c * dt;
B_d = B_c * dt;
g_d = g_vec * dt;

fprintf('State-space model created:\n');
fprintf('  States (nx): %d\n', nx);
fprintf('  Inputs (nu): %d\n', nu);
fprintf('  Sample time: %.4f s\n\n', dt);

%% ============================================================
% 3) LQR CONTROLLER DESIGN WITH INTEGRAL ACTION
%% ============================================================
% We augment the system with integral states for position/orientation
% tracking to eliminate steady-state error.
%
% Augmented state: x_aug = [x; z_int]
% where z_int = integral of [px_err; pz_err; theta_err]

ny = 3;  % outputs we track: px, pz, theta

% Output matrix (what we're tracking)
C_y = [1 0 0 0 0 0;   % px
       0 1 0 0 0 0;   % pz
       0 0 1 0 0 0];  % theta

% Augmented system for LQR with integral action
A_aug = [A_c,            zeros(nx, ny);
         -C_y,           zeros(ny, ny)];
     
B_aug = [B_c;
         zeros(ny, nu)];

% LQR weight matrices
% Q: penalize state deviations
%    [px, pz, theta, vx, vz, omega, int_px, int_pz, int_theta]
Q_state = diag([100,    ... % px position error
                500,    ... % pz height error (important!)
                300,    ... % theta pitch error (important!)
                10,     ... % vx velocity error
                50,     ... % vz velocity error  
                30]);   ... % omega pitch rate error

Q_int = diag([50,      ... % integral of px error
              200,     ... % integral of pz error (height)
              150]);   ... % integral of theta error

Q = blkdiag(Q_state, Q_int);

% R: penalize control effort
%    [f1x, f1z, f2x, f2z]
R = diag([0.001,   ... % front leg horizontal force
          0.0005,  ... % front leg vertical force
          0.001,   ... % rear leg horizontal force
          0.0005]);... % rear leg vertical force

% Compute LQR gain
[K_lqr_aug, ~, ~] = lqr(A_aug, B_aug, Q, R);

% Extract gains for original states and integral states
K_x = K_lqr_aug(:, 1:nx);      % 4 x 6
K_i = K_lqr_aug(:, nx+1:end);  % 4 x 3

fprintf('LQR Controller designed:\n');
fprintf('  State gain K_x: [%d x %d]\n', size(K_x, 1), size(K_x, 2));
fprintf('  Integral gain K_i: [%d x %d]\n\n', size(K_i, 1), size(K_i, 2));

%% ============================================================
% 4) BASELINE PD CONTROLLER DESIGN
%% ============================================================
% Simple PD control on position and orientation
% This represents a "traditional" approach without optimal control

% PD gains (tuned for reasonable performance)
Kp_pd = diag([300,    ... % px
              800,    ... % pz
              500]);  ... % theta

Kd_pd = diag([100,    ... % vx
              200,    ... % vz
              80]);   ... % omega

fprintf('Baseline PD Controller designed\n\n');

%% ============================================================
% 5) SIMULATION SETUP
%% ============================================================

T_sim = 5.0;           % simulation duration [s]
t = (0:dt:T_sim)';
N = length(t);

% Initial conditions
x0 = [0;              % px [m]
      params.h_nom;   % pz [m] - start at nominal height
      0;              % theta [rad]
      0;              % vx [m/s]
      0;              % vz [m/s]
      0];             % omega [rad/s]

%% ============================================================
% 6) REFERENCE TRAJECTORY GENERATION
%% ============================================================
% Create a challenging trajectory that demonstrates LQR advantages:
% - Step changes in height and pitch
% - Ramp velocity (acceleration)
% - Periodic motion

% Preallocate reference
x_ref = zeros(N, nx);

for k = 1:N
    tk = t(k);
    
    % Horizontal position: accelerate forward
    if tk < 1.0
        px_ref = 0;
        vx_ref = 0;
    elseif tk < 3.0
        % Linear acceleration phase
        px_ref = 0.3 * (tk - 1.0)^2;  % quadratic position
        vx_ref = 0.6 * (tk - 1.0);     % linear velocity
    else
        % Constant velocity phase
        px_ref = 0.3 * 4 + 1.2 * (tk - 3.0);
        vx_ref = 1.2;
    end
    
    % Vertical position: step change at t=2s
    if tk < 2.0
        pz_ref = params.h_nom;
    elseif tk < 2.5
        % Smooth transition using cosine interpolation
        pz_ref = params.h_nom + 0.05 * (1 - cos(pi * (tk - 2.0) / 0.5)) / 2;
    else
        pz_ref = params.h_nom + 0.05;  % 5cm higher
    end
    
    % Pitch: periodic motion to simulate terrain anticipation
    if tk > 1.5
        theta_ref = 0.1 * sin(2 * pi * 0.5 * (tk - 1.5));  % ~0.1 rad = 5.7 deg
    else
        theta_ref = 0;
    end
    
    % Store reference
    x_ref(k, :) = [px_ref, pz_ref, theta_ref, vx_ref, 0, 0];
end

fprintf('Reference trajectory generated:\n');
fprintf('  Duration: %.1f s\n', T_sim);
fprintf('  Max forward velocity: %.1f m/s\n', max(x_ref(:,4)));
fprintf('  Height change: %.2f m\n', max(x_ref(:,2)) - min(x_ref(:,2)));
fprintf('  Max pitch: %.1f deg\n\n', rad2deg(max(abs(x_ref(:,3)))));

%% ============================================================
% 7) SIMULATION: LQR CONTROLLER
%% ============================================================

fprintf('Running LQR simulation...\n');

% State and control logs
x_lqr = zeros(N, nx);
u_lqr = zeros(N, nu);
x_lqr(1, :) = x0';

% Integral state
z_int = zeros(ny, 1);

% Equilibrium forces (to counteract gravity)
% At equilibrium: f1z + f2z = m*g, split equally
f_eq = [0; params.m * params.g / 2; 0; params.m * params.g / 2];

for k = 1:(N-1)
    % Current state
    x_curr = x_lqr(k, :)';
    
    % Reference at current time
    ref_curr = x_ref(k, :)';
    
    % Error
    e_state = x_curr - ref_curr;
    e_track = C_y * e_state;  % tracking error for integral
    
    % Update integral (with anti-windup clamp)
    z_int = z_int + e_track * dt;
    z_int = max(min(z_int, 1.0), -1.0);  % clamp integral
    
    % LQR control law (deviation from equilibrium)
    u_delta = -K_x * e_state - K_i * z_int;
    
    % Total control: equilibrium + deviation
    u = f_eq + u_delta;
    
    % Apply force constraints
    u = apply_force_constraints(u, params);
    
    % Store control
    u_lqr(k, :) = u';
    
    % Discrete-time dynamics with gravity
    x_next = A_d * x_curr + B_d * u + g_d;
    
    % Simple ground constraint (can't go below ground)
    if x_next(2) < 0.1
        x_next(2) = 0.1;
        x_next(5) = max(x_next(5), 0);  % no downward velocity
    end
    
    x_lqr(k+1, :) = x_next';
end

fprintf('  LQR simulation complete.\n');

%% ============================================================
% 8) SIMULATION: BASELINE PD CONTROLLER
%% ============================================================

fprintf('Running PD baseline simulation...\n');

% State and control logs
x_pd = zeros(N, nx);
u_pd = zeros(N, nu);
x_pd(1, :) = x0';

for k = 1:(N-1)
    % Current state
    x_curr = x_pd(k, :)';
    
    % Reference at current time
    ref_curr = x_ref(k, :)';
    
    % Position/orientation errors
    e_pos = ref_curr(1:3) - x_curr(1:3);  % [px; pz; theta] error
    e_vel = ref_curr(4:6) - x_curr(4:6);  % [vx; vz; omega] error
    
    % PD control law: compute desired body forces/torque
    F_des = Kp_pd * e_pos + Kd_pd * e_vel;  % [Fx; Fz; Tau]
    
    % Add gravity compensation
    F_des(2) = F_des(2) + params.m * params.g;
    
    % Distribute forces to legs (simple equal distribution)
    % Fx = f1x + f2x  ->  f1x = f2x = Fx/2
    % Fz = f1z + f2z  ->  f1z = f2z = Fz/2
    % Tau = L*f1z - L*f2z  ->  f1z = (Fz + Tau/L)/2, f2z = (Fz - Tau/L)/2
    
    f1x = F_des(1) / 2;
    f2x = F_des(1) / 2;
    f1z = (F_des(2) + F_des(3) / params.L) / 2;
    f2z = (F_des(2) - F_des(3) / params.L) / 2;
    
    u = [f1x; f1z; f2x; f2z];
    
    % Apply force constraints
    u = apply_force_constraints(u, params);
    
    % Store control
    u_pd(k, :) = u';
    
    % Discrete-time dynamics with gravity
    x_next = A_d * x_curr + B_d * u + g_d;
    
    % Ground constraint
    if x_next(2) < 0.1
        x_next(2) = 0.1;
        x_next(5) = max(x_next(5), 0);
    end
    
    x_pd(k+1, :) = x_next';
end

fprintf('  PD simulation complete.\n\n');

%% ============================================================
% 9) DISTURBANCE REJECTION TEST
%% ============================================================

fprintf('Running disturbance rejection test...\n');

% Reset states
x_lqr_dist = zeros(N, nx);
x_pd_dist = zeros(N, nx);
x_lqr_dist(1, :) = x0';
x_pd_dist(1, :) = x0';

% Integral state for LQR
z_int_dist = zeros(ny, 1);

% Disturbance: impulse force at t = 2.5s
t_disturb = 2.5;
F_disturb = [50; 0; 0; 0; 0; 5];  % 50N horizontal push + pitch moment

% Use constant reference for cleaner comparison
x_ref_const = repmat([0, params.h_nom, 0, 0, 0, 0], N, 1);

for k = 1:(N-1)
    % Apply disturbance
    disturb = zeros(nx, 1);
    if abs(t(k) - t_disturb) < dt
        disturb = F_disturb * dt;
    end
    
    % --- LQR ---
    x_curr = x_lqr_dist(k, :)';
    ref_curr = x_ref_const(k, :)';
    e_state = x_curr - ref_curr;
    e_track = C_y * e_state;
    z_int_dist = z_int_dist + e_track * dt;
    z_int_dist = max(min(z_int_dist, 1.0), -1.0);
    u_delta = -K_x * e_state - K_i * z_int_dist;
    u = f_eq + u_delta;
    u = apply_force_constraints(u, params);
    x_next = A_d * x_curr + B_d * u + g_d + disturb;
    if x_next(2) < 0.1; x_next(2) = 0.1; x_next(5) = max(x_next(5), 0); end
    x_lqr_dist(k+1, :) = x_next';
    
    % --- PD ---
    x_curr = x_pd_dist(k, :)';
    e_pos = ref_curr(1:3) - x_curr(1:3);
    e_vel = ref_curr(4:6) - x_curr(4:6);
    F_des = Kp_pd * e_pos + Kd_pd * e_vel;
    F_des(2) = F_des(2) + params.m * params.g;
    f1x = F_des(1) / 2;
    f2x = F_des(1) / 2;
    f1z = (F_des(2) + F_des(3) / params.L) / 2;
    f2z = (F_des(2) - F_des(3) / params.L) / 2;
    u = apply_force_constraints([f1x; f1z; f2x; f2z], params);
    x_next = A_d * x_curr + B_d * u + g_d + disturb;
    if x_next(2) < 0.1; x_next(2) = 0.1; x_next(5) = max(x_next(5), 0); end
    x_pd_dist(k+1, :) = x_next';
end

fprintf('  Disturbance test complete.\n\n');

%% ============================================================
% 10) PERFORMANCE METRICS
%% ============================================================

fprintf('================= PERFORMANCE METRICS =================\n\n');

% --- Trajectory Tracking Metrics ---
fprintf('--- TRAJECTORY TRACKING ---\n');

% Position errors
e_px_lqr = x_ref(:,1) - x_lqr(:,1);
e_pz_lqr = x_ref(:,2) - x_lqr(:,2);
e_th_lqr = x_ref(:,3) - x_lqr(:,3);

e_px_pd = x_ref(:,1) - x_pd(:,1);
e_pz_pd = x_ref(:,2) - x_pd(:,2);
e_th_pd = x_ref(:,3) - x_pd(:,3);

% RMS errors
rms_px_lqr = sqrt(mean(e_px_lqr.^2));
rms_pz_lqr = sqrt(mean(e_pz_lqr.^2));
rms_th_lqr = sqrt(mean(e_th_lqr.^2));

rms_px_pd = sqrt(mean(e_px_pd.^2));
rms_pz_pd = sqrt(mean(e_pz_pd.^2));
rms_th_pd = sqrt(mean(e_th_pd.^2));

fprintf('RMS Position Error (px):\n');
fprintf('  LQR: %.4f m,  PD: %.4f m  (%.1f%% improvement)\n', ...
    rms_px_lqr, rms_px_pd, 100*(rms_px_pd - rms_px_lqr)/rms_px_pd);

fprintf('RMS Height Error (pz):\n');
fprintf('  LQR: %.4f m,  PD: %.4f m  (%.1f%% improvement)\n', ...
    rms_pz_lqr, rms_pz_pd, 100*(rms_pz_pd - rms_pz_lqr)/rms_pz_pd);

fprintf('RMS Pitch Error (theta):\n');
fprintf('  LQR: %.4f rad (%.2f deg),  PD: %.4f rad (%.2f deg)  (%.1f%% improvement)\n', ...
    rms_th_lqr, rad2deg(rms_th_lqr), rms_th_pd, rad2deg(rms_th_pd), ...
    100*(rms_th_pd - rms_th_lqr)/rms_th_pd);

% Max errors
fprintf('\nMax Absolute Errors:\n');
fprintf('  px:    LQR: %.4f m,    PD: %.4f m\n', max(abs(e_px_lqr)), max(abs(e_px_pd)));
fprintf('  pz:    LQR: %.4f m,    PD: %.4f m\n', max(abs(e_pz_lqr)), max(abs(e_pz_pd)));
fprintf('  theta: LQR: %.4f rad,  PD: %.4f rad\n', max(abs(e_th_lqr)), max(abs(e_th_pd)));

% --- Control Effort ---
fprintf('\n--- CONTROL EFFORT ---\n');
effort_lqr = sum(sum(u_lqr.^2)) * dt;
effort_pd = sum(sum(u_pd.^2)) * dt;
fprintf('Total Force-Squared Integral:\n');
fprintf('  LQR: %.2e N^2*s,  PD: %.2e N^2*s  (%.1f%% reduction)\n', ...
    effort_lqr, effort_pd, 100*(effort_pd - effort_lqr)/effort_pd);

% --- Disturbance Rejection ---
fprintf('\n--- DISTURBANCE REJECTION ---\n');

% Find settling time after disturbance (within 2cm of reference)
idx_disturb = find(t >= t_disturb, 1);
tol = 0.02;  % 2 cm

% LQR settling
pz_err_lqr = abs(x_lqr_dist(idx_disturb:end, 2) - params.h_nom);
idx_settle_lqr = find(pz_err_lqr < tol, 1);
if isempty(idx_settle_lqr)
    Ts_lqr = NaN;
else
    Ts_lqr = idx_settle_lqr * dt;
end

% PD settling  
pz_err_pd = abs(x_pd_dist(idx_disturb:end, 2) - params.h_nom);
idx_settle_pd = find(pz_err_pd < tol, 1);
if isempty(idx_settle_pd)
    Ts_pd = NaN;
else
    Ts_pd = idx_settle_pd * dt;
end

fprintf('Settling Time (height, 2cm tolerance):\n');
fprintf('  LQR: %.3f s,  PD: %.3f s\n', Ts_lqr, Ts_pd);

% Max deviation after disturbance
max_dev_lqr = max(abs(x_lqr_dist(idx_disturb:end, 1:3) - x_ref_const(idx_disturb:end, 1:3)));
max_dev_pd = max(abs(x_pd_dist(idx_disturb:end, 1:3) - x_ref_const(idx_disturb:end, 1:3)));

fprintf('Max Deviation After Disturbance:\n');
fprintf('  px:    LQR: %.4f m,    PD: %.4f m\n', max_dev_lqr(1), max_dev_pd(1));
fprintf('  pz:    LQR: %.4f m,    PD: %.4f m\n', max_dev_lqr(2), max_dev_pd(2));
fprintf('  theta: LQR: %.4f rad,  PD: %.4f rad\n', max_dev_lqr(3), max_dev_pd(3));

fprintf('\n=====================================================\n');

%% ============================================================
% 11) VISUALIZATION
%% ============================================================

% Figure 1: Trajectory Tracking Comparison
figure('Name', 'Trajectory Tracking Comparison', 'Position', [100, 100, 1200, 800]);

subplot(3, 2, 1);
plot(t, x_ref(:,1), 'k--', 'LineWidth', 1.5); hold on;
plot(t, x_lqr(:,1), 'b-', 'LineWidth', 1.2);
plot(t, x_pd(:,1), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('p_x [m]');
title('Horizontal Position Tracking');
legend('Reference', 'LQR', 'PD', 'Location', 'northwest');
grid on;

subplot(3, 2, 2);
plot(t, x_ref(:,4), 'k--', 'LineWidth', 1.5); hold on;
plot(t, x_lqr(:,4), 'b-', 'LineWidth', 1.2);
plot(t, x_pd(:,4), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('v_x [m/s]');
title('Horizontal Velocity Tracking');
legend('Reference', 'LQR', 'PD', 'Location', 'northwest');
grid on;

subplot(3, 2, 3);
plot(t, x_ref(:,2), 'k--', 'LineWidth', 1.5); hold on;
plot(t, x_lqr(:,2), 'b-', 'LineWidth', 1.2);
plot(t, x_pd(:,2), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('p_z [m]');
title('Height Tracking');
legend('Reference', 'LQR', 'PD', 'Location', 'northwest');
grid on;

subplot(3, 2, 4);
plot(t, x_lqr(:,5), 'b-', 'LineWidth', 1.2); hold on;
plot(t, x_pd(:,5), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('v_z [m/s]');
title('Vertical Velocity');
legend('LQR', 'PD', 'Location', 'northeast');
grid on;

subplot(3, 2, 5);
plot(t, rad2deg(x_ref(:,3)), 'k--', 'LineWidth', 1.5); hold on;
plot(t, rad2deg(x_lqr(:,3)), 'b-', 'LineWidth', 1.2);
plot(t, rad2deg(x_pd(:,3)), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('\theta [deg]');
title('Pitch Angle Tracking');
legend('Reference', 'LQR', 'PD', 'Location', 'northwest');
grid on;

subplot(3, 2, 6);
plot(t, rad2deg(x_lqr(:,6)), 'b-', 'LineWidth', 1.2); hold on;
plot(t, rad2deg(x_pd(:,6)), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('\omega [deg/s]');
title('Pitch Rate');
legend('LQR', 'PD', 'Location', 'northeast');
grid on;

sgtitle('Trajectory Tracking: LQR vs PD Control', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 2: Control Inputs
figure('Name', 'Control Inputs', 'Position', [150, 150, 1200, 600]);

subplot(2, 2, 1);
plot(t(1:end-1), u_lqr(1:end-1, 2), 'b-', 'LineWidth', 1); hold on;
plot(t(1:end-1), u_pd(1:end-1, 2), 'r-', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('f_{1z} [N]');
title('Front Leg Vertical Force');
legend('LQR', 'PD');
grid on;

subplot(2, 2, 2);
plot(t(1:end-1), u_lqr(1:end-1, 4), 'b-', 'LineWidth', 1); hold on;
plot(t(1:end-1), u_pd(1:end-1, 4), 'r-', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('f_{2z} [N]');
title('Rear Leg Vertical Force');
legend('LQR', 'PD');
grid on;

subplot(2, 2, 3);
plot(t(1:end-1), u_lqr(1:end-1, 1), 'b-', 'LineWidth', 1); hold on;
plot(t(1:end-1), u_pd(1:end-1, 1), 'r-', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('f_{1x} [N]');
title('Front Leg Horizontal Force');
legend('LQR', 'PD');
grid on;

subplot(2, 2, 4);
plot(t(1:end-1), u_lqr(1:end-1, 3), 'b-', 'LineWidth', 1); hold on;
plot(t(1:end-1), u_pd(1:end-1, 3), 'r-', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('f_{2x} [N]');
title('Rear Leg Horizontal Force');
legend('LQR', 'PD');
grid on;

sgtitle('Ground Reaction Forces: LQR vs PD Control', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 3: Disturbance Rejection
figure('Name', 'Disturbance Rejection', 'Position', [200, 200, 1000, 600]);

subplot(2, 2, 1);
plot(t, x_lqr_dist(:,1), 'b-', 'LineWidth', 1.2); hold on;
plot(t, x_pd_dist(:,1), 'r-', 'LineWidth', 1.2);
xline(t_disturb, 'k--', 'Disturbance', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('p_x [m]');
title('Horizontal Position');
legend('LQR', 'PD', 'Location', 'northwest');
grid on;

subplot(2, 2, 2);
plot(t, x_lqr_dist(:,2), 'b-', 'LineWidth', 1.2); hold on;
plot(t, x_pd_dist(:,2), 'r-', 'LineWidth', 1.2);
yline(params.h_nom, 'k--', 'Reference');
xline(t_disturb, 'k--', 'Disturbance', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('p_z [m]');
title('Height');
legend('LQR', 'PD', 'Location', 'southeast');
grid on;

subplot(2, 2, 3);
plot(t, rad2deg(x_lqr_dist(:,3)), 'b-', 'LineWidth', 1.2); hold on;
plot(t, rad2deg(x_pd_dist(:,3)), 'r-', 'LineWidth', 1.2);
yline(0, 'k--', 'Reference');
xline(t_disturb, 'k--', 'Disturbance', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('\theta [deg]');
title('Pitch Angle');
legend('LQR', 'PD', 'Location', 'northeast');
grid on;

subplot(2, 2, 4);
plot(t, x_lqr_dist(:,4), 'b-', 'LineWidth', 1.2); hold on;
plot(t, x_pd_dist(:,4), 'r-', 'LineWidth', 1.2);
xline(t_disturb, 'k--', 'Disturbance', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('v_x [m/s]');
title('Horizontal Velocity');
legend('LQR', 'PD', 'Location', 'northeast');
grid on;

sgtitle('Disturbance Rejection: LQR vs PD Control', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 4: Tracking Errors
figure('Name', 'Tracking Errors', 'Position', [250, 250, 1000, 400]);

subplot(1, 3, 1);
plot(t, e_px_lqr*100, 'b-', 'LineWidth', 1.2); hold on;
plot(t, e_px_pd*100, 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Error [cm]');
title('Position Error (p_x)');
legend('LQR', 'PD');
grid on;

subplot(1, 3, 2);
plot(t, e_pz_lqr*100, 'b-', 'LineWidth', 1.2); hold on;
plot(t, e_pz_pd*100, 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Error [cm]');
title('Height Error (p_z)');
legend('LQR', 'PD');
grid on;

subplot(1, 3, 3);
plot(t, rad2deg(e_th_lqr), 'b-', 'LineWidth', 1.2); hold on;
plot(t, rad2deg(e_th_pd), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Error [deg]');
title('Pitch Error (\theta)');
legend('LQR', 'PD');
grid on;

sgtitle('Tracking Errors Over Time', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\nSimulation complete. Figures generated.\n');

%% ============================================================
% HELPER FUNCTION: Apply Force Constraints
%% ============================================================

function u = apply_force_constraints(u, params)
    % Apply physical constraints to ground reaction forces
    % u = [f1x; f1z; f2x; f2z]
    
    % Vertical force limits
    u(2) = max(params.f_min, min(params.f_max, u(2)));  % f1z
    u(4) = max(params.f_min, min(params.f_max, u(4)));  % f2z
    
    % Friction cone constraints: |fx| <= mu * fz
    max_f1x = params.mu * u(2);
    max_f2x = params.mu * u(4);
    
    u(1) = max(-max_f1x, min(max_f1x, u(1)));  % f1x
    u(3) = max(-max_f2x, min(max_f2x, u(3)));  % f2x
end
