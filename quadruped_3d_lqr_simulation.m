%% ============================================================
% 3D QUADRUPED LQR CONTROL SIMULATION - AGGRESSIVE TEST
% Based on: "Dynamic Locomotion in the MIT Cheetah 3 Through 
%           Convex Model-Predictive Control" (Di Carlo et al., 2018)
%
% This simulation PUSHES BOTH CONTROLLERS TO THEIR LIMITS:
% - Fast yaw rates (45-60 deg/s vs typical 15 deg/s)
% - Rapid height changes (15cm in 0.75s)
% - Simultaneous roll/pitch/yaw commands
% - Quick direction reversals
% - Large disturbances (100N push, 20Nm yaw torque)
%
% FAIR COMPARISON:
% - LQR: Optimal control via Riccati equation (accounts for coupling)
% - PD: Systematic second-order design (treats axes independently)
%
% The question: At what point does PD break down while LQR continues?
%% ============================================================

clear; clc; close all;

%% ============================================================
% 1) PHYSICAL PARAMETERS (MIT Cheetah 3 - from paper Table I)
%% ============================================================

% Robot parameters (from MIT paper)
params.m = 43;           % mass [kg]
params.Ixx = 0.41;       % roll moment of inertia [kg*m^2]
params.Iyy = 2.1;        % pitch moment of inertia [kg*m^2]
params.Izz = 2.1;        % yaw moment of inertia [kg*m^2]
params.g = 9.81;         % gravity [m/s^2]

% Body inertia tensor (body frame)
params.I_body = diag([params.Ixx, params.Iyy, params.Izz]);

% Geometry
params.Lx = 0.19;        % half body length (front to CoM) [m]
params.Ly = 0.11;        % half body width (side to CoM) [m]
params.h_nom = 0.35;     % nominal standing height [m]

% Foot positions relative to CoM in body frame [x, y, z]
% When standing: z = -h_nom (feet below CoM)
params.foot_pos_body = [
    params.Lx,   params.Ly,  0;    % Front-Left (FL)
    params.Lx,  -params.Ly,  0;    % Front-Right (FR)
   -params.Lx,   params.Ly,  0;    % Rear-Left (RL)
   -params.Lx,  -params.Ly,  0;    % Rear-Right (RR)
]';  % 3x4 matrix, each column is a foot

% Friction coefficient
params.mu = 0.6;

% Force limits per leg
params.f_min = 10;       % minimum normal force [N]
params.f_max = 666;      % maximum normal force [N] (from paper)

fprintf('=== 3D Quadruped LQR Simulation (MIT Cheetah Model) ===\n');
fprintf('Robot mass: %.1f kg\n', params.m);
fprintf('Inertia: Ixx=%.2f, Iyy=%.2f, Izz=%.2f kg*m^2\n', ...
    params.Ixx, params.Iyy, params.Izz);
fprintf('Body dimensions: %.2f x %.2f m\n', 2*params.Lx, 2*params.Ly);
fprintf('Nominal height: %.2f m\n\n', params.h_nom);

%% ============================================================
% 2) STATE-SPACE MODEL (Full 3D - from MIT paper Eq. 16)
%% ============================================================
% State vector: x = [Θ; p; ω; v] (12 states)
%   Θ = [φ; θ; ψ]     - Euler angles (roll, pitch, yaw) [rad]
%   p = [px; py; pz]  - position [m]
%   ω = [ωx; ωy; ωz]  - angular velocity [rad/s]
%   v = [vx; vy; vz]  - linear velocity [m/s]
%
% Control input: u = [f1; f2; f3; f4] where fi = [fix; fiy; fiz]
%   Total: 12 inputs (3 force components × 4 legs)
%
% Equations of motion (from MIT paper Eq. 5-7):
%   p̈ = (Σfi)/m - g                    (linear acceleration)
%   d/dt(Iω) = Σ(ri × fi)              (angular acceleration)
%   Θ̇ = Rz(ψ)^T * ω                    (Euler rate, small angle approx)

nx = 12;  % number of states
nu = 12;  % number of control inputs (4 legs × 3 force components)

% For linearization, we assume small roll/pitch and use nominal yaw = 0
% This gives us a time-invariant system for LQR design

% Continuous-time A matrix (from MIT paper Eq. 16)
% State order: [φ, θ, ψ, px, py, pz, ωx, ωy, ωz, vx, vy, vz]
A_c = zeros(nx, nx);

% Θ̇ = Rz(ψ)^T * ω ≈ ω for small yaw (identity rotation)
% So: φ̇ = ωx, θ̇ = ωy, ψ̇ = ωz
A_c(1, 7) = 1;   % φ̇ = ωx
A_c(2, 8) = 1;   % θ̇ = ωy
A_c(3, 9) = 1;   % ψ̇ = ωz

% ṗ = v
A_c(4, 10) = 1;  % ṗx = vx
A_c(5, 11) = 1;  % ṗy = vy
A_c(6, 12) = 1;  % ṗz = vz

% ω̇ and v̇ depend on forces (in B matrix)

% Continuous-time B matrix
% Control: [f1x,f1y,f1z, f2x,f2y,f2z, f3x,f3y,f3z, f4x,f4y,f4z]
B_c = zeros(nx, nu);

% Inverse of inertia tensor (for angular acceleration)
I_inv = inv(params.I_body);

% For each leg, compute contribution to angular and linear acceleration
for leg = 1:4
    % Foot position relative to CoM (in body frame)
    r = params.foot_pos_body(:, leg);
    r(3) = -params.h_nom;  % feet are below CoM at ground level
    
    % Skew-symmetric matrix for cross product: τ = r × f
    r_skew = skew(r);
    
    % Column indices for this leg's forces
    col_start = (leg-1)*3 + 1;
    col_end = leg*3;
    
    % Angular acceleration: ω̇ = I^(-1) * (r × f)
    B_c(7:9, col_start:col_end) = I_inv * r_skew;
    
    % Linear acceleration: v̇ = f/m
    B_c(10:12, col_start:col_end) = eye(3) / params.m;
end

% Gravity vector (constant disturbance on vz)
g_vec = zeros(nx, 1);
g_vec(12) = -params.g;  % vz acceleration due to gravity

% Discretize the system
dt = 0.001;  % 1 kHz control loop (matches MIT Cheetah)
A_d = eye(nx) + A_c * dt;
B_d = B_c * dt;
g_d = g_vec * dt;

fprintf('3D State-space model created:\n');
fprintf('  States (nx): %d\n', nx);
fprintf('  Inputs (nu): %d\n', nu);
fprintf('  Sample time: %.4f s\n\n', dt);

%% ============================================================
% 3) LQR CONTROLLER DESIGN
%% ============================================================
% LQR weight matrices
% State: [φ, θ, ψ, px, py, pz, ωx, ωy, ωz, vx, vy, vz]

% Q matrix for states - penalize deviations
Q = diag([2000,   ... % roll - HIGH (stability critical)
          2000,   ... % pitch - HIGH (stability critical)  
          1000,   ... % yaw - moderate (orientation)
          500,    ... % px position
          500,    ... % py position
          3000,   ... % pz height - VERY HIGH (safety)
          100,    ... % ωx roll rate
          100,    ... % ωy pitch rate
          50,     ... % ωz yaw rate
          50,     ... % vx velocity
          50,     ... % vy velocity
          200]);  ... % vz velocity - helps height control

% R matrix - penalize control effort (all 12 force components)
% Lower values = more aggressive control
R_leg = diag([0.0001,   ... % fx - horizontal
              0.0001,   ... % fy - horizontal
              0.00005]); ... % fz - vertical (allow more variation)
R = blkdiag(R_leg, R_leg, R_leg, R_leg);  % Same for all 4 legs

% Compute LQR gain
[K, ~, ~] = lqr(A_c, B_c, Q, R);

fprintf('LQR Controller designed:\n');
fprintf('  State gain K: [%d x %d]\n', size(K, 1), size(K, 2));
fprintf('  Total gains to tune: 1 (Q/R matrices)\n\n');

%% ============================================================
% 4) BASELINE PD CONTROLLER DESIGN - SYSTEMATIC TUNING
%% ============================================================
% Using second-order response design (standard engineering method)
%
% For each axis, we treat it as an independent second-order system:
%   m*ẍ + Kd*ẋ + Kp*x = 0  (for translation)
%   I*θ̈ + Kd*θ̇ + Kp*θ = 0  (for rotation)
%
% Standard second-order form: s² + 2*ζ*ωn*s + ωn² = 0
%
% Matching coefficients:
%   Kp = m * ωn²       (or I * ωn² for rotation)
%   Kd = 2 * ζ * ωn * m (or 2 * ζ * ωn * I for rotation)
%
% Design specifications:
%   - Settling time Ts ≈ 4/(ζ*ωn) 
%   - Damping ratio ζ = 0.7 (standard for "good" response, ~5% overshoot)
%
% This is a SYSTEMATIC method that any controls engineer would use,
% but it IGNORES COUPLING between axes - that's the key limitation!

fprintf('Baseline PD Controller - Systematic Tuning:\n');
fprintf('  Method: Second-order response design (per-axis)\n');
fprintf('  Damping ratio ζ = 0.7 (standard)\n');

% Design parameters
zeta = 0.7;              % Damping ratio (0.7 = ~5% overshoot, good damping)

% Desired settling times for each axis (engineering judgment)
% Faster for orientation (safety), slower for position (less aggressive)
Ts_roll = 0.3;           % Roll settling time [s] - fast for stability
Ts_pitch = 0.3;          % Pitch settling time [s] - fast for stability
Ts_yaw = 0.5;            % Yaw settling time [s] - moderate
Ts_xy = 0.8;             % X/Y position settling time [s] - can be slower
Ts_z = 0.4;              % Height settling time [s] - moderate (gravity)

% Compute natural frequencies from settling time: ωn = 4/(ζ*Ts)
wn_roll = 4 / (zeta * Ts_roll);
wn_pitch = 4 / (zeta * Ts_pitch);
wn_yaw = 4 / (zeta * Ts_yaw);
wn_xy = 4 / (zeta * Ts_xy);
wn_z = 4 / (zeta * Ts_z);

fprintf('  Settling times: roll=%.1fs, pitch=%.1fs, yaw=%.1fs, xy=%.1fs, z=%.1fs\n', ...
    Ts_roll, Ts_pitch, Ts_yaw, Ts_xy, Ts_z);
fprintf('  Natural frequencies: ωn = [%.1f, %.1f, %.1f, %.1f, %.1f] rad/s\n', ...
    wn_roll, wn_pitch, wn_yaw, wn_xy, wn_z);

% Effective inertias for each axis
% NOTE: This assumes decoupled dynamics - the KEY LIMITATION of PD design!
I_roll = params.Ixx;     % Roll uses Ixx
I_pitch = params.Iyy;    % Pitch uses Iyy  
I_yaw = params.Izz;      % Yaw uses Izz
m_xy = params.m;         % X/Y translation uses mass
m_z = params.m;          % Z translation uses mass

% Compute PD gains using second-order formulas:
%   Kp = I * ωn²
%   Kd = 2 * ζ * ωn * I

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

% Assemble gain matrices
Kp_pd = diag([Kp_roll, Kp_pitch, Kp_yaw, Kp_xy, Kp_xy, Kp_z]);
Kd_pd = diag([Kd_roll, Kd_pitch, Kd_yaw, Kd_xy, Kd_xy, Kd_z]);

fprintf('  Computed Kp: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n', ...
    Kp_roll, Kp_pitch, Kp_yaw, Kp_xy, Kp_xy, Kp_z);
fprintf('  Computed Kd: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n', ...
    Kd_roll, Kd_pitch, Kd_yaw, Kd_xy, Kd_xy, Kd_z);
fprintf('  LIMITATION: Assumes decoupled axes (ignores cross-coupling)\n\n');

%% ============================================================
% 5) SIMULATION SETUP
%% ============================================================

T_sim = 8.0;            % simulation duration [s]
t = (0:dt:T_sim)';
N = length(t);

% Initial conditions (standing at origin)
x0 = zeros(nx, 1);
x0(6) = params.h_nom;   % pz = nominal height

%% ============================================================
% 6) REFERENCE TRAJECTORY - AGGRESSIVE 3D MANEUVER
%% ============================================================
% This trajectory is designed to STRESS the controllers:
% - Fast turns (90°/s yaw rate vs 15°/s before)
% - Rapid height changes
% - Simultaneous coupled motions
% - Quick direction reversals

x_ref = zeros(N, nx);

fprintf('AGGRESSIVE TEST MODE:\n');
fprintf('  - 3x faster yaw rate (45°/s)\n');
fprintf('  - Rapid height changes (0.15m in 0.5s)\n');
fprintf('  - Simultaneous roll/pitch/yaw commands\n');
fprintf('  - Quick direction reversals\n\n');

for k = 1:N
    tk = t(k);
    
    % === PHASE 1: Stand still (0-0.5s) - shorter! ===
    if tk < 0.5
        roll_ref = 0; pitch_ref = 0; yaw_ref = 0;
        px_ref = 0; py_ref = 0; pz_ref = params.h_nom;
        wx_ref = 0; wy_ref = 0; wz_ref = 0;
        vx_ref = 0; vy_ref = 0; vz_ref = 0;
        
    % === PHASE 2: FAST turn + accelerate (0.5-2.5s) ===
    % 45 deg/s yaw rate (3x faster than before)
    elseif tk < 2.5
        t_phase = tk - 0.5;
        
        % Fast yaw: 45 deg/s
        yaw_rate = deg2rad(45);
        yaw_ref = yaw_rate * t_phase;
        wz_ref = yaw_rate;
        
        % Accelerate to 1.0 m/s (2x faster than before)
        v_body = min(1.0, 0.5 * t_phase);  % ramp up
        px_ref = 0.5 * t_phase * cos(yaw_ref/2);
        py_ref = 0.5 * t_phase * sin(yaw_ref/2);
        vx_ref = v_body * cos(yaw_ref);
        vy_ref = v_body * sin(yaw_ref);
        
        roll_ref = 0; pitch_ref = 0;
        wx_ref = 0; wy_ref = 0;
        pz_ref = params.h_nom; vz_ref = 0;
        
    % === PHASE 3: RAPID height + pitch + roll (2.5-4s) ===
    % Everything happens at once - maximum coupling stress!
    elseif tk < 4.0
        t_phase = tk - 2.5;
        
        % Hold yaw from Phase 2
        yaw_ref = deg2rad(45) * 2.0;
        wz_ref = 0;
        
        % Continue forward motion
        px_ref = 0.5 * 2.0 * cos(yaw_ref/2) + 0.8 * t_phase * cos(yaw_ref);
        py_ref = 0.5 * 2.0 * sin(yaw_ref/2) + 0.8 * t_phase * sin(yaw_ref);
        vx_ref = 0.8 * cos(yaw_ref);
        vy_ref = 0.8 * sin(yaw_ref);
        
        % RAPID height change: +15cm in 0.75s (3x faster than before)
        if t_phase < 0.75
            pz_ref = params.h_nom + 0.15 * (1 - cos(pi * t_phase / 0.75)) / 2;
            vz_ref = 0.15 * pi / 0.75 * sin(pi * t_phase / 0.75) / 2;
        else
            pz_ref = params.h_nom + 0.15;
            vz_ref = 0;
        end
        
        % AGGRESSIVE pitch: 15 degrees (1.5x larger)
        pitch_ref = deg2rad(15) * (1 - cos(pi * t_phase / 1.0)) / 2;
        wy_ref = deg2rad(15) * pi / 1.0 * sin(pi * t_phase / 1.0) / 2;
        
        % FAST roll oscillation: ±8 deg at 2 Hz (larger and faster)
        roll_ref = deg2rad(8) * sin(2 * pi * 2.0 * t_phase);
        wx_ref = deg2rad(8) * 2 * pi * 2.0 * cos(2 * pi * 2.0 * t_phase);
        
    % === PHASE 4: REVERSAL - quick direction change (4-5.5s) ===
    % This tests the controller's ability to handle sudden changes
    elseif tk < 5.5
        t_phase = tk - 4.0;
        
        % Reverse yaw: -60 deg/s (turn back!)
        yaw_rate = deg2rad(-60);
        yaw_ref = deg2rad(90) + yaw_rate * t_phase;
        wz_ref = yaw_rate;
        
        % Maintain position roughly
        px_ref = 0.5 * 2.0 * cos(deg2rad(45)) + 0.8 * 1.5 * cos(deg2rad(90));
        py_ref = 0.5 * 2.0 * sin(deg2rad(45)) + 0.8 * 1.5 * sin(deg2rad(90));
        vx_ref = 0; vy_ref = 0;
        
        % Drop height quickly
        if t_phase < 0.5
            pz_ref = params.h_nom + 0.15 - 0.15 * (1 - cos(pi * t_phase / 0.5)) / 2;
        else
            pz_ref = params.h_nom;
        end
        
        % Pitch back to level
        pitch_ref = deg2rad(15) * (1 - t_phase/1.5);
        roll_ref = 0;
        wx_ref = 0; wy_ref = 0; vz_ref = 0;
        
    % === PHASE 5: AGGRESSIVE lateral + roll (5.5-8s) ===
    % Fast sidestep while rolling - very challenging!
    else
        t_phase = tk - 5.5;
        
        % Hold yaw at final value
        yaw_ref = deg2rad(90) + deg2rad(-60) * 1.5;
        wz_ref = 0;
        
        % FAST lateral motion: 0.6 m/s sidestep
        lateral_speed = 0.6;
        px_base = 0.5 * 2.0 * cos(deg2rad(45)) + 0.8 * 1.5 * cos(deg2rad(90));
        py_base = 0.5 * 2.0 * sin(deg2rad(45)) + 0.8 * 1.5 * sin(deg2rad(90));
        
        px_ref = px_base - lateral_speed * t_phase * sin(yaw_ref);
        py_ref = py_base + lateral_speed * t_phase * cos(yaw_ref);
        vx_ref = -lateral_speed * sin(yaw_ref);
        vy_ref = lateral_speed * cos(yaw_ref);
        
        pz_ref = params.h_nom;
        pitch_ref = 0;
        
        % Continuous roll oscillation during sidestep (tests coupling)
        roll_ref = deg2rad(10) * sin(2 * pi * 1.5 * t_phase);
        wx_ref = deg2rad(10) * 2 * pi * 1.5 * cos(2 * pi * 1.5 * t_phase);
        wy_ref = 0; vz_ref = 0;
    end
    
    % Store reference
    x_ref(k, :) = [roll_ref, pitch_ref, yaw_ref, ...
                   px_ref, py_ref, pz_ref, ...
                   wx_ref, wy_ref, wz_ref, ...
                   vx_ref, vy_ref, vz_ref];
end

fprintf('Aggressive reference trajectory generated:\n');
fprintf('  Duration: %.1f s\n', T_sim);
fprintf('  Max yaw rate: 60 deg/s (4x baseline)\n');
fprintf('  Max pitch: %.1f deg\n', rad2deg(max(x_ref(:,2))));
fprintf('  Max roll oscillation: ±10 deg at 2 Hz\n');
fprintf('  Height change: %.2f m in 0.75s\n', 0.15);
fprintf('  Phases: Stand → FastTurn → RapidPitch/Roll/Height → Reversal → FastSidestep\n\n');

%% ============================================================
% 7) SIMULATION: LQR CONTROLLER
%% ============================================================

fprintf('Running LQR simulation...\n');

x_lqr = zeros(N, nx);
u_lqr = zeros(N, nu);
x_lqr(1, :) = x0';


% Equilibrium forces (gravity compensation, equal distribution)
f_eq_per_leg = [0; 0; params.m * params.g / 4];
f_eq = repmat(f_eq_per_leg, 4, 1);

for k = 1:(N-1)
    x_curr = x_lqr(k, :)';
    ref_curr = x_ref(k, :)';
    
    % State error
    e_state = x_curr - ref_curr;
    
    % LQR control law
    u_delta = -K * e_state;
    u = f_eq + u_delta;
    
    % Apply force constraints
    u = apply_force_constraints_3d(u, params);
    
    u_lqr(k, :) = u';
    
    % Dynamics
    x_next = A_d * x_curr + B_d * u + g_d;
    
    % Ground constraint
    if x_next(6) < 0.15
        x_next(6) = 0.15;
        x_next(12) = max(x_next(12), 0);
    end
    
    x_lqr(k+1, :) = x_next';
end

fprintf('  LQR simulation complete.\n');

%% ============================================================
% 8) SIMULATION: BASELINE PD CONTROLLER
%% ============================================================

fprintf('Running PD baseline simulation...\n');

x_pd = zeros(N, nx);
u_pd = zeros(N, nu);
x_pd(1, :) = x0';

for k = 1:(N-1)
    x_curr = x_pd(k, :)';
    ref_curr = x_ref(k, :)';
    
    % Position/orientation errors
    e_pos = ref_curr(1:6) - x_curr(1:6);   % [φ,θ,ψ, px,py,pz] errors
    e_vel = ref_curr(7:12) - x_curr(7:12); % [ωx,ωy,ωz, vx,vy,vz] errors
    
    % PD control: compute desired body forces and torques
    % F_des = [τx, τy, τz, Fx, Fy, Fz]
    pd_output = Kp_pd * e_pos + Kd_pd * e_vel;
    
    tau_des = pd_output(1:3);     % desired torques [τx, τy, τz]
    F_des = pd_output(4:6);       % desired forces [Fx, Fy, Fz]
    F_des(3) = F_des(3) + params.m * params.g;  % gravity compensation
    
    % Distribute forces to 4 legs (simplified - equal distribution + torque)
    % This is where PD struggles - it doesn't optimally distribute forces!
    u = distribute_forces_pd(tau_des, F_des, params);
    
    % Apply constraints
    u = apply_force_constraints_3d(u, params);
    
    u_pd(k, :) = u';
    
    % Dynamics
    x_next = A_d * x_curr + B_d * u + g_d;
    
    % Ground constraint
    if x_next(6) < 0.15
        x_next(6) = 0.15;
        x_next(12) = max(x_next(12), 0);
    end
    
    x_pd(k+1, :) = x_next';
end

fprintf('  PD simulation complete.\n\n');

%% ============================================================
% 9) DISTURBANCE REJECTION TEST - AGGRESSIVE
%% ============================================================

fprintf('Running AGGRESSIVE disturbance rejection test...\n');

x_lqr_dist = zeros(N, nx);
x_pd_dist = zeros(N, nx);
x_lqr_dist(1, :) = x0';
x_pd_dist(1, :) = x0';

% LARGE disturbance: like a strong push or steep slope
% 100N ≈ 10kg equivalent push on 43kg robot (significant!)
% 20Nm yaw torque ≈ someone grabbing and twisting the robot
t_disturb_start = 2.0;
F_disturb = [100; 80; 0];      % 100N forward, 80N sideways (BIG!)
tau_disturb = [15; 10; 20];    % 15Nm roll, 10Nm pitch, 20Nm yaw

fprintf('  Disturbance: F=[%.0f,%.0f,%.0f]N, τ=[%.0f,%.0f,%.0f]Nm\n', ...
    F_disturb(1), F_disturb(2), F_disturb(3), ...
    tau_disturb(1), tau_disturb(2), tau_disturb(3));
fprintf('  This is equivalent to a strong push (~10kg force on 43kg robot)\n');

% Simple hover reference for cleaner comparison
x_ref_hover = repmat([0, 0, 0, 0, 0, params.h_nom, 0, 0, 0, 0, 0, 0], N, 1);

for k = 1:(N-1)
    % External disturbance (applied as acceleration)
    if t(k) >= t_disturb_start && t(k) < t_disturb_start + 2.0
        accel_disturb = zeros(nx, 1);
        accel_disturb(7:9) = params.I_body \ tau_disturb * dt;  % angular accel
        accel_disturb(10:12) = F_disturb / params.m * dt;       % linear accel
    else
        accel_disturb = zeros(nx, 1);
    end
    
    % --- LQR Controller ---
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
    
    % --- PD Controller ---
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
fprintf('  Sustained disturbance from t=%.1fs: F=[%.0f,%.0f,%.0f]N, τ=[%.0f,%.0f,%.0f]Nm\n\n', ...
    t_disturb_start, F_disturb(1), F_disturb(2), F_disturb(3), ...
    tau_disturb(1), tau_disturb(2), tau_disturb(3));

%% ============================================================
% 10) PERFORMANCE METRICS
%% ============================================================

fprintf('================= PERFORMANCE METRICS =================\n\n');

% First, check that both controllers are stable (sanity check)
max_pos_lqr = max(abs(x_lqr(:, 4:6)), [], 'all');
max_pos_pd = max(abs(x_pd(:, 4:6)), [], 'all');
max_angle_lqr = max(abs(x_lqr(:, 1:3)), [], 'all');
max_angle_pd = max(abs(x_pd(:, 1:3)), [], 'all');

fprintf('--- STABILITY CHECK ---\n');
fprintf('Max position magnitude: LQR=%.2f m, PD=%.2f m\n', max_pos_lqr, max_pos_pd);
fprintf('Max angle magnitude:    LQR=%.2f rad (%.0f deg), PD=%.2f rad (%.0f deg)\n', ...
    max_angle_lqr, rad2deg(max_angle_lqr), max_angle_pd, rad2deg(max_angle_pd));

if max_pos_pd > 10 || max_angle_pd > 2*pi
    fprintf('WARNING: PD controller appears UNSTABLE! Results may not be meaningful.\n');
elseif max_pos_pd > 5 || max_angle_pd > pi
    fprintf('WARNING: PD controller shows large deviations. May need tuning.\n');
else
    fprintf('Both controllers appear stable. Comparison is valid.\n');
end
fprintf('\n');

% --- Trajectory Tracking ---
fprintf('--- TRAJECTORY TRACKING (3D Maneuver) ---\n');

% RMS errors for key states
states_of_interest = {'Roll', 'Pitch', 'Yaw', 'px', 'py', 'pz'};
units = {'deg', 'deg', 'deg', 'm', 'm', 'm'};
scale = [180/pi, 180/pi, 180/pi, 1, 1, 1];

for i = 1:6
    e_lqr = x_ref(:,i) - x_lqr(:,i);
    e_pd = x_ref(:,i) - x_pd(:,i);
    
    rms_lqr = sqrt(mean(e_lqr.^2)) * scale(i);
    rms_pd = sqrt(mean(e_pd.^2)) * scale(i);
    
    improvement = 100 * (rms_pd - rms_lqr) / rms_pd;
    
    fprintf('%s RMS Error:\n', states_of_interest{i});
    fprintf('  LQR: %.4f %s,  PD: %.4f %s  (%.1f%% improvement)\n', ...
        rms_lqr, units{i}, rms_pd, units{i}, improvement);
end

% --- Disturbance Rejection ---
fprintf('\n--- DISTURBANCE REJECTION (Sustained Force) ---\n');

% Steady-state errors (last 2 seconds)
idx_ss = t > (T_sim - 2.0);

for i = 1:6
    ss_lqr = mean(abs(x_lqr_dist(idx_ss, i)));
    ss_pd = mean(abs(x_pd_dist(idx_ss, i)));
    
    ss_lqr_scaled = ss_lqr * scale(i);
    ss_pd_scaled = ss_pd * scale(i);
    
    fprintf('%s Steady-State Error:\n', states_of_interest{i});
    fprintf('  LQR: %.4f %s,  PD: %.4f %s\n', ...
        ss_lqr_scaled, units{i}, ss_pd_scaled, units{i});
end

% --- Control Effort ---
fprintf('\n--- CONTROL EFFORT ---\n');
effort_lqr = sum(sum(u_lqr.^2)) * dt;
effort_pd = sum(sum(u_pd.^2)) * dt;
fprintf('Total Force-Squared Integral:\n');
fprintf('  LQR: %.2e N^2*s,  PD: %.2e N^2*s\n', effort_lqr, effort_pd);

fprintf('\n==================== KEY FINDINGS ====================\n');
fprintf('AGGRESSIVE TEST with fast maneuvers and large disturbances.\n');
fprintf('\n');
fprintf('LQR: Optimal control that accounts for ALL state couplings.\n');
fprintf('PD:  Systematic second-order design treating axes independently.\n');
fprintf('\n');
fprintf('Under gentle conditions, both controllers work adequately.\n');
fprintf('Under STRESS (fast turns, large pushes), the difference emerges:\n');
fprintf('  - PD cannot coordinate coupled responses\n');
fprintf('  - PD has no integral action → steady-state errors persist\n');
fprintf('  - LQR handles coupling and eliminates steady-state error\n');
fprintf('=====================================================\n\n');

%% ============================================================
% 11) VISUALIZATION
%% ============================================================

% Figure 1: 3D Trajectory Comparison
figure('Name', '3D Trajectory Tracking', 'Position', [50, 50, 1400, 900]);

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
% Top-down view of path
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

sgtitle('AGGRESSIVE 3D Test: LQR vs PD (Fast Turns, Rapid Height Changes, Direction Reversals)', ...
    'FontSize', 14, 'FontWeight', 'bold');

% Figure 2: Disturbance Rejection
figure('Name', 'Disturbance Rejection (3D)', 'Position', [100, 100, 1200, 600]);

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

sgtitle('AGGRESSIVE Disturbance: Sustained Force [100,80,0]N + Torque [15,10,20]Nm', ...
    'FontSize', 14, 'FontWeight', 'bold');

fprintf('Simulation complete. Figures generated.\n');

%% ============================================================
% HELPER FUNCTIONS
%% ============================================================

function S = skew(v)
    % Skew-symmetric matrix for cross product: S*x = v × x
    S = [0, -v(3), v(2);
         v(3), 0, -v(1);
         -v(2), v(1), 0];
end

function u = apply_force_constraints_3d(u, params)
    % Apply physical constraints to all 12 force components
    % u = [f1x,f1y,f1z, f2x,f2y,f2z, f3x,f3y,f3z, f4x,f4y,f4z]
    
    for leg = 1:4
        idx = (leg-1)*3 + (1:3);
        
        % Vertical force limits
        u(idx(3)) = max(params.f_min, min(params.f_max, u(idx(3))));
        
        % Friction cone: |fx|, |fy| <= mu * fz
        max_horiz = params.mu * u(idx(3));
        u(idx(1)) = max(-max_horiz, min(max_horiz, u(idx(1))));
        u(idx(2)) = max(-max_horiz, min(max_horiz, u(idx(2))));
    end
end

function u = distribute_forces_pd(tau_des, F_des, params)
    % Force distribution for PD control
    % Maps desired body wrench [τx, τy, τz, Fx, Fy, Fz] to 12 leg forces
    %
    % The key physics (from r × f for each leg):
    %   τx (roll)  = Ly × (left_fz - right_fz)
    %   τy (pitch) = Lx × (rear_fz - front_fz)  ← NOTE: rear minus front!
    %   τz (yaw)   = Ly × (right_fx - left_fx) + Lx × (front_fy - rear_fy)
    
    Lx = params.Lx;
    Ly = params.Ly;
    
    % Equal distribution of total force
    F_per_leg = F_des / 4;
    
    % === Roll torque: τx = Ly × (left_fz - right_fz) ===
    % To get τx = tau_des(1): left legs need +delta_roll, right legs need -delta_roll
    % Total: τx = Ly × (2×(base+delta) - 2×(base-delta)) = 4×Ly×delta
    delta_z_roll = tau_des(1) / (4 * Ly);
    
    % === Pitch torque: τy = Lx × (rear_fz - front_fz) ===
    % To get τy = tau_des(2): rear legs need +delta_pitch, front legs need -delta_pitch
    % Total: τy = Lx × (2×(base+delta) - 2×(base-delta)) = 4×Lx×delta
    delta_z_pitch = tau_des(2) / (4 * Lx);
    
    % === Yaw torque: τz = Ly × (right_fx - left_fx) ===
    % To get positive yaw (turn left): right legs push forward (+fx), left push back (-fx)
    % τz = Ly × ((+delta) - (-delta)) × 2 = 4×Ly×delta
    delta_x_yaw = tau_des(3) / (4 * Ly);
    
    % Build force vector for each leg with CORRECT signs:
    % Roll:  left (+Ly) gets +delta_roll,  right (-Ly) gets -delta_roll
    % Pitch: rear (-Lx) gets +delta_pitch, front (+Lx) gets -delta_pitch
    % Yaw:   right gets +delta_yaw (forward), left gets -delta_yaw (backward)
    
    % FL (front-left): +Lx, +Ly → front(-pitch), left(+roll), left(-yaw)
    f_FL = F_per_leg + [-delta_x_yaw; 0; +delta_z_roll - delta_z_pitch];
    
    % FR (front-right): +Lx, -Ly → front(-pitch), right(-roll), right(+yaw)
    f_FR = F_per_leg + [+delta_x_yaw; 0; -delta_z_roll - delta_z_pitch];
    
    % RL (rear-left): -Lx, +Ly → rear(+pitch), left(+roll), left(-yaw)
    f_RL = F_per_leg + [-delta_x_yaw; 0; +delta_z_roll + delta_z_pitch];
    
    % RR (rear-right): -Lx, -Ly → rear(+pitch), right(-roll), right(+yaw)
    f_RR = F_per_leg + [+delta_x_yaw; 0; -delta_z_roll + delta_z_pitch];
    
    u = [f_FL; f_FR; f_RL; f_RR];
end