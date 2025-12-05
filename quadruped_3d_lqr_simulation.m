clear; clc; close all;

%% Physical Parameters From MIT Cheetah 3 Paper

% Init
% Mass and gravity
params.m = 43;
params.g = 9.81;

% Intertia
params.Ixx = 0.41;
params.Iyy = 2.1;
params.Izz = 2.1;
params.I_body = diag([params.Ixx, params.Iyy, params.Izz]);

% Geometry
params.Lx = 0.19;
params.Ly = 0.11;
params.h_nom = 0.35;

% Foot positions (body corners)
params.foot_pos_body = [
    params.Lx,   params.Ly,  0;    % Front-Left (FL)
    params.Lx,  -params.Ly,  0;    % Front-Right (FR)
   -params.Lx,   params.Ly,  0;    % Rear-Left (RL)
   -params.Lx,  -params.Ly,  0;    % Rear-Right (RR)
]';

% Friction coefficient
params.mu = 0.6;

% Force limits per leg
params.f_min = 10;
params.f_max = 666;

%% State-Space Representation

% num states & num inputs
nx = 12;
nu = 12;

% Init and fill in A matrix
A_c = zeros(nx, nx);
A_c(1, 7) = 1;
A_c(2, 8) = 1;
A_c(3, 9) = 1;
A_c(4, 10) = 1;
A_c(5, 11) = 1;
A_c(6, 12) = 1;

% Init and fill in B matrix
B_c = zeros(nx, nu);
I_inv = inv(params.I_body);

% For each leg...
for leg = 1:4
    % Leg indices for B matrix
    col_start = (leg-1)*3 + 1;
    col_end = leg*3;

    % Get foot position wrt body frame
    r = params.foot_pos_body(:, leg);
    r(3) = -params.h_nom;
    
    % Get ang and linear accels
    r_skew = skew(r);
    B_c(7:9, col_start:col_end) = I_inv * r_skew;
    B_c(10:12, col_start:col_end) = eye(3) / params.m;
end

% Create grav vector
g_vec = zeros(nx, 1);
g_vec(12) = -params.g;

% Make the system time discrete
dt = 0.001;
A_d = eye(nx) + A_c * dt;
B_d = B_c * dt;
g_d = g_vec * dt;

%% LQR Design

% Fill in Q matrix
Q = diag([2000,   ... % roll
          2000,   ... % pitch
          1000,   ... % yaw
          500,    ... % px
          500,    ... % py
          3000,   ... % pz
          100,    ... % ωx
          100,    ... % ωy
          50,     ... % ωz
          50,     ... % vx
          50,     ... % vy
          200]);  ... % vz

% Fill in R matrix for all 4 legs
R_leg = diag([0.0001,    ... % fx
              0.0001,    ... % fy
              0.00005]); ... % fz
R = blkdiag(R_leg, R_leg, R_leg, R_leg);

% Get optimal gain matrix
[K, ~, ~] = lqr(A_c, B_c, Q, R);

%% PD Design

% Desired damping and time domain specs
zeta = 0.7;
Ts_roll = 0.3;
Ts_pitch = 0.3;
Ts_yaw = 0.5;
Ts_xy = 0.8;
Ts_z = 0.4;

% Get natural frequencies from chosen specs
wn_roll = 4 / (zeta * Ts_roll);
wn_pitch = 4 / (zeta * Ts_pitch);
wn_yaw = 4 / (zeta * Ts_yaw);
wn_xy = 4 / (zeta * Ts_xy);
wn_z = 4 / (zeta * Ts_z);

% Grab mass/intertia info to calculate gains
I_roll = params.Ixx;
I_pitch = params.Iyy;
I_yaw = params.Izz;
m_xy = params.m;
m_z = params.m;

% Calculate PD gains
Kp_roll = I_roll * wn_roll^2;
Kd_roll = 2 * zeta * wn_roll * I_roll;
Kp_pitch = I_pitch * wn_pitch^2;
Kd_pitch = 2 * zeta * wn_pitch * I_pitch;
Kp_yaw = I_yaw * wn_yaw^2;
Kd_yaw = 2 * zeta * wn_yaw * I_yaw;
Kp_xy = m_xy * wn_xy^2;
Kd_xy = 2 * zeta * wn_xy * m_xy;
Kp_z = m_z * wn_z^2;
Kd_z = 2 * zeta * wn_z * m_z;

% Create gain matrices
Kp_pd = diag([Kp_roll, Kp_pitch, Kp_yaw, Kp_xy, Kp_xy, Kp_z]);
Kd_pd = diag([Kd_roll, Kd_pitch, Kd_yaw, Kd_xy, Kd_xy, Kd_z]);

%% Simulation Setup

% Simulation timing
T_sim = 8.0;
t = (0:dt:T_sim)';
N = length(t);

% Initial conditions
x0 = zeros(nx, 1);
x0(6) = params.h_nom;

%% Create the reference trajectory

% Init trajectory ref states
x_ref = zeros(N, nx);

% For every dt...
for k = 1:N

    % Grab current time
    tk = t(k);
    
    % Trajectory phase 1 - stand still
    if tk < 0.5
        roll_ref = 0; pitch_ref = 0; yaw_ref = 0;
        px_ref = 0; py_ref = 0; pz_ref = params.h_nom;
        wx_ref = 0; wy_ref = 0; wz_ref = 0;
        vx_ref = 0; vy_ref = 0; vz_ref = 0;
        
    % Trajectory phase 2 - turn quickly and accelerate
    elseif tk < 2.5
        % Time relative to start of phase
        t_phase = tk - 0.5;
        
        % Yaw ref
        yaw_rate = deg2rad(45);
        yaw_ref = yaw_rate * t_phase;
        wz_ref = yaw_rate;
        
        % Acceleration
        v_body = min(1.0, 0.5 * t_phase);
        px_ref = 0.5 * t_phase * cos(yaw_ref/2);
        py_ref = 0.5 * t_phase * sin(yaw_ref/2);
        vx_ref = v_body * cos(yaw_ref);
        vy_ref = v_body * sin(yaw_ref);
        
        % Unchanged
        roll_ref = 0; pitch_ref = 0;
        wx_ref = 0; wy_ref = 0;
        pz_ref = params.h_nom; vz_ref = 0;
        
    % Trajectory phase 3 - z translations and pitch/roll
    elseif tk < 4.0
        % Time relative to start of phase
        t_phase = tk - 2.5;
        
        % Keep prev yaw
        yaw_ref = deg2rad(45) * 2.0;
        wz_ref = 0;
        
        % Keep moving forward
        px_ref = 0.5 * 2.0 * cos(yaw_ref/2) + 0.8 * t_phase * cos(yaw_ref);
        py_ref = 0.5 * 2.0 * sin(yaw_ref/2) + 0.8 * t_phase * sin(yaw_ref);
        vx_ref = 0.8 * cos(yaw_ref);
        vy_ref = 0.8 * sin(yaw_ref);
        
        % Z translations
        if t_phase < 0.75
            pz_ref = params.h_nom + 0.15 * (1 - cos(pi * t_phase / 0.75)) / 2;
            vz_ref = 0.15 * pi / 0.75 * sin(pi * t_phase / 0.75) / 2;
        else
            pz_ref = params.h_nom + 0.15;
            vz_ref = 0;
        end
        
        % Large pitch
        pitch_ref = deg2rad(15) * (1 - cos(pi * t_phase / 1.0)) / 2;
        wy_ref = deg2rad(15) * pi / 1.0 * sin(pi * t_phase / 1.0) / 2;
        
        % Large roll
        roll_ref = deg2rad(8) * sin(2 * pi * 2.0 * t_phase);
        wx_ref = deg2rad(8) * 2 * pi * 2.0 * cos(2 * pi * 2.0 * t_phase);
        
    % Trajectory phase 4 - direction change
    elseif tk < 5.5
        % Time relative to start of phase
        t_phase = tk - 4.0;
        
        % Yaw back
        yaw_rate = deg2rad(-60);
        yaw_ref = deg2rad(90) + yaw_rate * t_phase;
        wz_ref = yaw_rate;
        
        % Keep position
        px_ref = 0.5 * 2.0 * cos(deg2rad(45)) + 0.8 * 1.5 * cos(deg2rad(90));
        py_ref = 0.5 * 2.0 * sin(deg2rad(45)) + 0.8 * 1.5 * sin(deg2rad(90));
        vx_ref = 0; vy_ref = 0;
        
        % Undo z translation
        if t_phase < 0.5
            pz_ref = params.h_nom + 0.15 - 0.15 * (1 - cos(pi * t_phase / 0.5)) / 2;
        else
            pz_ref = params.h_nom;
        end
        
        % Pitch back
        pitch_ref = deg2rad(15) * (1 - t_phase/1.5);
        roll_ref = 0;
        wx_ref = 0; wy_ref = 0; vz_ref = 0;
        
    % Trajectory phase 5 - lateral movement and roll
    else
        % Time relative to start of phase
        t_phase = tk - 5.5;
        
        % Keep yaw
        yaw_ref = deg2rad(90) + deg2rad(-60) * 1.5;
        wz_ref = 0;
        
        % Give lateral movement
        lateral_speed = 0.6;
        px_base = 0.5 * 2.0 * cos(deg2rad(45)) + 0.8 * 1.5 * cos(deg2rad(90));
        py_base = 0.5 * 2.0 * sin(deg2rad(45)) + 0.8 * 1.5 * sin(deg2rad(90));
        
        px_ref = px_base - lateral_speed * t_phase * sin(yaw_ref);
        py_ref = py_base + lateral_speed * t_phase * cos(yaw_ref);
        vx_ref = -lateral_speed * sin(yaw_ref);
        vy_ref = lateral_speed * cos(yaw_ref);
        
        pz_ref = params.h_nom;
        pitch_ref = 0;
        
        % Give roll
        roll_ref = deg2rad(10) * sin(2 * pi * 1.5 * t_phase);
        wx_ref = deg2rad(10) * 2 * pi * 1.5 * cos(2 * pi * 1.5 * t_phase);
        wy_ref = 0; vz_ref = 0;
    end
    
    % Store ref state
    x_ref(k, :) = [roll_ref, pitch_ref, yaw_ref, ...
                   px_ref, py_ref, pz_ref, ...
                   wx_ref, wy_ref, wz_ref, ...
                   vx_ref, vy_ref, vz_ref];
end

fprintf('Reference Trajectory Specs:\n');
fprintf('  Duration: %.1f s\n', T_sim);
fprintf('  Max yaw rate: 60 deg/s\n');
fprintf('  Max pitch: %.1f deg\n', rad2deg(max(x_ref(:,2))));
fprintf('  Roll pattern: ±10 deg @ 2 Hz\n');
fprintf('  Height change: %.2f m\n\n', 0.15);

%% Run sim with LQR controller

fprintf('Running LQR simulation...\n');

% Init state & control matrices
x_lqr = zeros(N, nx);
u_lqr = zeros(N, nu);
x_lqr(1, :) = x0';

% Init required equilibrium forces
f_eq_per_leg = [0; 0; params.m * params.g / 4];
f_eq = repmat(f_eq_per_leg, 4, 1);

% For each timestep...
for k = 1:(N-1)

    % Get current state and reference trajectory state
    x_curr = x_lqr(k, :)';
    ref_curr = x_ref(k, :)';
    
    % Get state error
    e_state = x_curr - ref_curr;
    
    % Apply LQR
    u_delta = -K * e_state;
    u = f_eq + u_delta;
    
    % Clamp force
    u = apply_force_constraints_3d(u, params);
    u_lqr(k, :) = u';
    
    % Update state
    x_next = A_d * x_curr + B_d * u + g_d;
    
    % Ground constraint
    if x_next(6) < 0.15
        x_next(6) = 0.15;
        x_next(12) = max(x_next(12), 0);
    end
    
    % Save next state
    x_lqr(k+1, :) = x_next';
end

fprintf('  LQR simulation complete.\n\n');

%% Run sim with PD controller

fprintf('Running PD simulation...\n');

% Init state & control matrices
x_pd = zeros(N, nx);
u_pd = zeros(N, nu);
x_pd(1, :) = x0';

% For each timestep...
for k = 1:(N-1)

    % Get current state and reference trajectory state
    x_curr = x_pd(k, :)';
    ref_curr = x_ref(k, :)';
    
    % Get position & orientation errors
    e_pos = ref_curr(1:6) - x_curr(1:6);
    e_vel = ref_curr(7:12) - x_curr(7:12);
    
    % Compute desired wrench
    pd_output = Kp_pd * e_pos + Kd_pd * e_vel;
    tau_des = pd_output(1:3);
    F_des = pd_output(4:6);
    F_des(3) = F_des(3) + params.m * params.g;
    
    % Distribute net body force to each leg
    u = distribute_forces_pd(tau_des, F_des, params);
    
    % Clamp forces
    u = apply_force_constraints_3d(u, params);
    u_pd(k, :) = u';
    
    % Update state
    x_next = A_d * x_curr + B_d * u + g_d;
    
    % Ground constraint
    if x_next(6) < 0.15
        x_next(6) = 0.15;
        x_next(12) = max(x_next(12), 0);
    end
    
    % Save next state
    x_pd(k+1, :) = x_next';
end

fprintf('  PD simulation complete.\n\n');

%% Disturbance rejection simulation
fprintf('Running disturbance rejection simulation...\n');

% Init states for each controller
x_lqr_dist = zeros(N, nx);
x_pd_dist = zeros(N, nx);
x_lqr_dist(1, :) = x0';
x_pd_dist(1, :) = x0';

% Init disturbances
t_disturb_start = 2.0;
F_disturb = [100; 80; 0];
tau_disturb = [15; 10; 20];
fprintf('  Disturbance: F = [%.0f, %.0f, %.0f]N, τ = [%.0f, %.0f, %.0f]Nm\n', ...
    F_disturb(1), F_disturb(2), F_disturb(3), ...
    tau_disturb(1), tau_disturb(2), tau_disturb(3));

% Stand still ref
x_ref_hover = repmat([0, 0, 0, 0, 0, params.h_nom, 0, 0, 0, 0, 0, 0], N, 1);

% For every timestep...
for k = 1:(N-1)

    % Apply disturbance when specified
    if t(k) >= t_disturb_start && t(k) < t_disturb_start + 2.0
        accel_disturb = zeros(nx, 1);
        accel_disturb(7:9) = params.I_body \ tau_disturb * dt;
        accel_disturb(10:12) = F_disturb / params.m * dt;
    else
        accel_disturb = zeros(nx, 1);
    end
    
    % Update LQR states
    x_curr = x_lqr_dist(k, :)';
    ref_curr = x_ref_hover(k, :)';
    e_state = x_curr - ref_curr; 
    u_delta = -K * e_state;
    u = f_eq + u_delta;
    u = apply_force_constraints_3d(u, params);
    x_next = A_d * x_curr + B_d * u + g_d + accel_disturb;
    if x_next(6) < 0.15
        x_next(6) = 0.15;
        x_next(12) = max(x_next(12), 0);
    end
    x_lqr_dist(k+1, :) = x_next';
    
    % Update PD states
    x_curr = x_pd_dist(k, :)';
    e_pos = ref_curr(1:6) - x_curr(1:6);
    e_vel = ref_curr(7:12) - x_curr(7:12); 
    pd_output = Kp_pd * e_pos + Kd_pd * e_vel;
    tau_des = pd_output(1:3);
    F_des = pd_output(4:6);
    F_des(3) = F_des(3) + params.m * params.g;
    u = distribute_forces_pd(tau_des, F_des, params);
    u = apply_force_constraints_3d(u, params);
    x_next = A_d * x_curr + B_d * u + g_d + accel_disturb;
    if x_next(6) < 0.15
        x_next(6) = 0.15;
        x_next(12) = max(x_next(12), 0);
    end
    x_pd_dist(k+1, :) = x_next';
end

fprintf('  Disturbance test complete.\n');

%% Print performance metrics

fprintf('== PERFORMANCE METRICS ==\n\n');

% Check for stability
max_pos_lqr = max(abs(x_lqr(:, 4:6)), [], 'all');
max_pos_pd = max(abs(x_pd(:, 4:6)), [], 'all');
max_angle_lqr = max(abs(x_lqr(:, 1:3)), [], 'all');
max_angle_pd = max(abs(x_pd(:, 1:3)), [], 'all');
fprintf('STABILITY CHECK\n');
fprintf('  Max position magnitude: LQR=%.2f m, PD=%.2f m\n', max_pos_lqr, max_pos_pd);
fprintf('  Max angle magnitude:    LQR=%.2f rad (%.0f deg), PD=%.2f rad (%.0f deg)\n', ...
    max_angle_lqr, rad2deg(max_angle_lqr), max_angle_pd, rad2deg(max_angle_pd));
if max_pos_pd > 10 || max_angle_pd > 2*pi
    fprintf('WARNING: PD controller is unstable.\n');
elseif max_pos_pd > 5 || max_angle_pd > pi
    fprintf('WARNING: PD controller may need tuning.\n');
else
    fprintf('Both controllers are stable.\n');
end

% Check trajectory tracking
fprintf('\nTRAJECTORY TRACKING\n');

% Get RMS errors for each pose state 
pose_states = {'Roll', 'Pitch', 'Yaw', 'px', 'py', 'pz'};
units = {'deg', 'deg', 'deg', 'm', 'm', 'm'};
scale = [180/pi, 180/pi, 180/pi, 1, 1, 1];
for i = 1:6
    e_lqr = x_ref(:,i) - x_lqr(:,i);
    e_pd = x_ref(:,i) - x_pd(:,i);
    rms_lqr = sqrt(mean(e_lqr.^2)) * scale(i);
    rms_pd = sqrt(mean(e_pd.^2)) * scale(i);
    improvement = 100 * (rms_pd - rms_lqr) / rms_pd;   
    fprintf('  %s RMS Error:\n', pose_states{i});
    fprintf('    LQR: %.4f %s,  PD: %.4f %s  (%.1f%% improvement)\n', ...
        rms_lqr, units{i}, rms_pd, units{i}, improvement);
end

% Check disturbance rejection
fprintf('\nDISTURBANCE REJECTION\n');

% Get steady-state errors for each pose
idx_ss = t > (T_sim - 2.0);
for i = 1:6
    ss_lqr = mean(abs(x_lqr_dist(idx_ss, i)));
    ss_pd = mean(abs(x_pd_dist(idx_ss, i)));
    ss_lqr_scaled = ss_lqr * scale(i);
    ss_pd_scaled = ss_pd * scale(i);
    fprintf('  %s Steady-State Error:\n', pose_states{i});
    fprintf('    LQR: %.4f %s,  PD: %.4f %s\n', ...
        ss_lqr_scaled, units{i}, ss_pd_scaled, units{i});
end

% Check control effort
fprintf('\nCONTROL EFFORT\n');
effort_lqr = sum(sum(u_lqr.^2)) * dt;
effort_pd = sum(sum(u_pd.^2)) * dt;
fprintf('  Total Force-Squared For All Time:\n');
fprintf('    LQR: %.2e N^2*s,  PD: %.2e N^2*s\n', effort_lqr, effort_pd);

%% Visualization

% Figure 1: Trajectory Tracking Tests
figure('Name', 'Trajectory Tracking Tests', 'Position', [50, 50, 1400, 900]);

% Row 1: Orientation
subplot(3, 4, 1);
plot(t, rad2deg(x_ref(:,1)), 'k--', 'LineWidth', 1.5); hold on;
plot(t, rad2deg(x_lqr(:,1)), 'b-', 'LineWidth', 1.2);
plot(t, rad2deg(x_pd(:,1)), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('\phi [deg]');
title('Roll Tracking');
legend('Ref', 'LQR', 'PD', 'Location', 'best');
grid on;

subplot(3, 4, 2);
plot(t, rad2deg(x_ref(:,2)), 'k--', 'LineWidth', 1.5); hold on;
plot(t, rad2deg(x_lqr(:,2)), 'b-', 'LineWidth', 1.2);
plot(t, rad2deg(x_pd(:,2)), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('\theta [deg]');
title('Pitch Tracking');
legend('Ref', 'LQR', 'PD', 'Location', 'best');
grid on;

subplot(3, 4, 3);
plot(t, rad2deg(x_ref(:,3)), 'k--', 'LineWidth', 1.5); hold on;
plot(t, rad2deg(x_lqr(:,3)), 'b-', 'LineWidth', 1.2);
plot(t, rad2deg(x_pd(:,3)), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('\psi [deg]');
title('Yaw Tracking');
legend('Ref', 'LQR', 'PD', 'Location', 'best');
grid on;

subplot(3, 4, 4);
plot3(x_ref(:,4), x_ref(:,5), x_ref(:,6), 'k--', 'LineWidth', 2); hold on;
plot3(x_lqr(:,4), x_lqr(:,5), x_lqr(:,6), 'b-', 'LineWidth', 1.5);
plot3(x_pd(:,4), x_pd(:,5), x_pd(:,6), 'r-', 'LineWidth', 1.5);
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
title('3D Path');
legend('Ref', 'LQR', 'PD');
grid on; view(45, 30);

% Row 2: Position
subplot(3, 4, 5);
plot(t, x_ref(:,4), 'k--', 'LineWidth', 1.5); hold on;
plot(t, x_lqr(:,4), 'b-', 'LineWidth', 1.2);
plot(t, x_pd(:,4), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('p_x [m]');
title('X Position Tracking');
legend('Ref', 'LQR', 'PD', 'Location', 'best');
grid on;

subplot(3, 4, 6);
plot(t, x_ref(:,5), 'k--', 'LineWidth', 1.5); hold on;
plot(t, x_lqr(:,5), 'b-', 'LineWidth', 1.2);
plot(t, x_pd(:,5), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('p_y [m]');
title('Y Position Tracking');
legend('Ref', 'LQR', 'PD', 'Location', 'best');
grid on;

subplot(3, 4, 7);
plot(t, x_ref(:,6), 'k--', 'LineWidth', 1.5); hold on;
plot(t, x_lqr(:,6), 'b-', 'LineWidth', 1.2);
plot(t, x_pd(:,6), 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('p_z [m]');
title('Height Tracking');
legend('Ref', 'LQR', 'PD', 'Location', 'best');
grid on;

subplot(3, 4, 8);
plot(x_ref(:,4), x_ref(:,5), 'k--', 'LineWidth', 2); hold on;
plot(x_lqr(:,4), x_lqr(:,5), 'b-', 'LineWidth', 1.5);
plot(x_pd(:,4), x_pd(:,5), 'r-', 'LineWidth', 1.5);
xlabel('x [m]'); ylabel('y [m]');
title('Path (Top View)');
legend('Ref', 'LQR', 'PD');
grid on; axis equal;

% Row 3: Tracking Errors
subplot(3, 4, 9);
e_roll_lqr = rad2deg(x_ref(:,1) - x_lqr(:,1));
e_roll_pd = rad2deg(x_ref(:,1) - x_pd(:,1));
plot(t, e_roll_lqr, 'b-', 'LineWidth', 1.2); hold on;
plot(t, e_roll_pd, 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Error [deg]');
title('Roll Error');
legend('LQR', 'PD');
grid on;

subplot(3, 4, 10);
e_pitch_lqr = rad2deg(x_ref(:,2) - x_lqr(:,2));
e_pitch_pd = rad2deg(x_ref(:,2) - x_pd(:,2));
plot(t, e_pitch_lqr, 'b-', 'LineWidth', 1.2); hold on;
plot(t, e_pitch_pd, 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Error [deg]');
title('Pitch Error');
legend('LQR', 'PD');
grid on;

subplot(3, 4, 11);
e_yaw_lqr = rad2deg(x_ref(:,3) - x_lqr(:,3));
e_yaw_pd = rad2deg(x_ref(:,3) - x_pd(:,3));
plot(t, e_yaw_lqr, 'b-', 'LineWidth', 1.2); hold on;
plot(t, e_yaw_pd, 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Error [deg]');
title('Yaw Error');
legend('LQR', 'PD');
grid on;

subplot(3, 4, 12);
e_height_lqr = (x_ref(:,6) - x_lqr(:,6)) * 100;
e_height_pd = (x_ref(:,6) - x_pd(:,6)) * 100;
plot(t, e_height_lqr, 'b-', 'LineWidth', 1.2); hold on;
plot(t, e_height_pd, 'r-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Error [cm]');
title('Height Error');
legend('LQR', 'PD');
grid on;

sgtitle('Trajectory Tracking Tests', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 2: Disturbance Rejection Tests
figure('Name', 'Disturbance Rejection Tests', 'Position', [100, 100, 1200, 600]);

subplot(2, 3, 1);
plot(t, rad2deg(x_lqr_dist(:,1)), 'b-', 'LineWidth', 1.2); hold on;
plot(t, rad2deg(x_pd_dist(:,1)), 'r-', 'LineWidth', 1.2);
xline(t_disturb_start, 'k--', 'Disturbance', 'LineWidth', 1);
yline(0, 'k:');
xlabel('Time [s]'); ylabel('\phi [deg]');
title('Roll Response');
legend('LQR', 'PD');
grid on;

subplot(2, 3, 2);
plot(t, rad2deg(x_lqr_dist(:,2)), 'b-', 'LineWidth', 1.2); hold on;
plot(t, rad2deg(x_pd_dist(:,2)), 'r-', 'LineWidth', 1.2);
xline(t_disturb_start, 'k--', 'Disturbance', 'LineWidth', 1);
yline(0, 'k:');
xlabel('Time [s]'); ylabel('\theta [deg]');
title('Pitch Response');
legend('LQR', 'PD');
grid on;

subplot(2, 3, 3);
plot(t, rad2deg(x_lqr_dist(:,3)), 'b-', 'LineWidth', 1.2); hold on;
plot(t, rad2deg(x_pd_dist(:,3)), 'r-', 'LineWidth', 1.2);
xline(t_disturb_start, 'k--', 'Disturbance', 'LineWidth', 1);
yline(0, 'k:');
xlabel('Time [s]'); ylabel('\psi [deg]');
title('Yaw Response');
legend('LQR', 'PD');
grid on;

subplot(2, 3, 4);
plot(t, x_lqr_dist(:,4)*100, 'b-', 'LineWidth', 1.2); hold on;
plot(t, x_pd_dist(:,4)*100, 'r-', 'LineWidth', 1.2);
xline(t_disturb_start, 'k--', 'Disturbance', 'LineWidth', 1);
yline(0, 'k:');
xlabel('Time [s]'); ylabel('p_x [cm]');
title('X Position Response');
legend('LQR', 'PD');
grid on;

subplot(2, 3, 5);
plot(t, x_lqr_dist(:,5)*100, 'b-', 'LineWidth', 1.2); hold on;
plot(t, x_pd_dist(:,5)*100, 'r-', 'LineWidth', 1.2);
xline(t_disturb_start, 'k--', 'Disturbance', 'LineWidth', 1);
yline(0, 'k:');
xlabel('Time [s]'); ylabel('p_y [cm]');
title('Y Position Response');
legend('LQR', 'PD');
grid on;

subplot(2, 3, 6);
plot(t, x_lqr_dist(:,6), 'b-', 'LineWidth', 1.2); hold on;
plot(t, x_pd_dist(:,6), 'r-', 'LineWidth', 1.2);
xline(t_disturb_start, 'k--', 'Disturbance', 'LineWidth', 1);
yline(params.h_nom, 'k:', 'Reference');
xlabel('Time [s]'); ylabel('p_z [m]');
title('Height Response');
legend('LQR', 'PD');
grid on;

sgtitle('Disturbance Rejection Tests', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('Figures generated.\n');

%% Helper functions

function S = skew(v)
    % Create a skew symmetric matrix
    S = [0, -v(3), v(2);
         v(3), 0, -v(1);
         -v(2), v(1), 0];
end

function u = apply_force_constraints_3d(u, params)
    % Apply physical constraints
    
    % For each leg...
    for leg = 1:4
        idx = (leg-1)*3 + (1:3);
        
        % Vertical force limits
        u(idx(3)) = max(params.f_min, min(params.f_max, u(idx(3))));
        
        % Friction force limits
        max_horiz = params.mu * u(idx(3));
        u(idx(1)) = max(-max_horiz, min(max_horiz, u(idx(1))));
        u(idx(2)) = max(-max_horiz, min(max_horiz, u(idx(2))));
    end
end

function u = distribute_forces_pd(tau_des, F_des, params)
    % Distribute desired wrench forces to each leg

    % Body params
    Lx = params.Lx;
    Ly = params.Ly;
    
    % Distribute force equally
    F_per_leg = F_des / 4;
    
    % Achieve roll torque by actuation in z
    delta_z_roll = tau_des(1) / (4 * Ly);
    
    % Achieve pitch torque by actuation in z
    delta_z_pitch = tau_des(2) / (4 * Lx);
    
    % Achieve yaw torque by actuation in x
    delta_x_yaw = tau_des(3) / (4 * Ly);
       
    % Add force deltas to each leg
    f_FL = F_per_leg + [-delta_x_yaw; 0; +delta_z_roll - delta_z_pitch];
    f_FR = F_per_leg + [+delta_x_yaw; 0; -delta_z_roll - delta_z_pitch];
    f_RL = F_per_leg + [-delta_x_yaw; 0; +delta_z_roll + delta_z_pitch];
    f_RR = F_per_leg + [+delta_x_yaw; 0; -delta_z_roll + delta_z_pitch];
    
    % Stack control input
    u = [f_FL; f_FR; f_RL; f_RR];
end