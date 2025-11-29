%% 1) Define model
m = 40;   % mass [kg]

% State: x = [px; py; vx; vy]
A = [0 0 1 0;
     0 0 0 1;
     0 0 0 0;
     0 0 0 0];

B = [0   0;
     0   0;
     1/m 0;
     0   1/m];

% Outputs we care about: y = [px; py]
Cy = [1 0 0 0;
      0 1 0 0];

nx = 4;       % states
ny = 2;       % outputs


%% 2) Build augmented system and COMPUTE LQR GAIN (before sim)

A_a = [A           zeros(nx,ny);
       -Cy         zeros(ny,ny)];
B_a = [B;
       zeros(ny,2)];

Qx = diag([1000 10 100 1]);
Qi = 200*eye(ny);
Q  = blkdiag(Qx, Qi);

R  = 0.001*eye(2);

K_a = lqr(A_a, B_a, Q, R);   % <-- gain computed here, once

Kx = K_a(:, 1:nx);           % 2x4
Ki = K_a(:, nx+1:end);       % 2x2


%% 3) Make a reference trajectory y_ref(t)

t = (0:0.001:10)';   % 0 to 10 s
N = length(t);

px_ref = 0.4 * t;               % forward
py_ref = 0.3 * t;    % weave

y_ref_all = [px_ref py_ref];    % for convenience


%% 4) SIMULATE the closed loop using the PRECOMPUTED Kx, Ki

dt = t(2) - t(1);

x = zeros(nx,1);    % [px; py; vx; vy]
z = zeros(ny,1);    % integral of error [ex; ey]

x_log = zeros(nx,N);
u_log = zeros(2,N);

for k = 1:N
    y_ref = y_ref_all(k,:).';    % [px_ref; py_ref]

    y = Cy * x;                  % current [px; py]
    e = y_ref - y;               % error

    z = z + e * dt;              % integrate error

    % USE GAINS HERE (already computed above)
    u = -Kx * x - Ki * z;        % [Fx; Fy]

    x_dot = A * x + B * u;
    x = x + x_dot * dt;

    x_log(:,k) = x;
    u_log(:,k) = u;
end


%% 5) Plot tracking
figure;
subplot(2,1,1);
plot(t, px_ref, 'k--', t, x_log(1,:), 'LineWidth', 1.3);
legend('p_x ref','p_x'); grid on;

subplot(2,1,2);
plot(t, py_ref, 'k--', t, x_log(2,:), 'LineWidth', 1.3);
legend('p_y ref','p_y'); grid on;

%% ============================================================
% VALIDITY METRICS FOR TRAJECTORY TRACKING
% Requires:
%   t        (Nx1) time
%   px, py   actual trajectories
%   px_ref, py_ref desired trajectories
%% ============================================================

fprintf("\n================= VALIDITY METRICS =================\n");

%% Tracking error
ex = px_ref - px;
ey = py_ref - py;

%% ---- Max absolute error ----
max_ex = max(abs(ex));
max_ey = max(abs(ey));

%% ---- RMS error ----
rms_ex = sqrt(mean(ex.^2));
rms_ey = sqrt(mean(ey.^2));

%% ---- Steady-state error (last 20% of simulation) ----
idx_ss = round(0.8*length(t)):length(t);

ess_px = mean(px_ref(idx_ss) - px(idx_ss));
ess_py = mean(py_ref(idx_ss) - py(idx_ss));

%% ---- Overshoot (relative overshoot in tracking) ----
% compute overshoot wrt final reference value
final_px = px_ref(end);
final_py = py_ref(end);

% guard against division by zero
if abs(final_px) > 1e-6
    OS_px = (max(px) - final_px) / abs(final_px) * 100;
else
    OS_px = max(abs(px));  % absolute overshoot
end

if abs(final_py) > 1e-6
    OS_py = (max(py) - final_py) / abs(final_py) * 100;
else
    OS_py = max(abs(py));
end

%% ---- Settling time (within ±2 cm band) ----
tol = 0.02;   % 2 cm tolerance

idx_px = find(abs(px - final_px) <= tol, 1, 'first');
idx_py = find(abs(py - final_py) <= tol, 1, 'first');

if isempty(idx_px); Ts_px = NaN; else; Ts_px = t(idx_px); end
if isempty(idx_py); Ts_py = NaN; else; Ts_py = t(idx_py); end

%% ---- Print results ----
fprintf("Max Error (x): %.4f m\n", max_ex);
fprintf("Max Error (y): %.4f m\n", max_ey);

fprintf("RMS Error (x): %.4f m\n", rms_ex);
fprintf("RMS Error (y): %.4f m\n", rms_ey);

fprintf("Steady-State Error (x): %.4f m\n", ess_px);
fprintf("Steady-State Error (y): %.4f m\n", ess_py);

fprintf("Overshoot (x): %.2f %% (or %.4f m if zero-ref)\n", OS_px, OS_px);
fprintf("Overshoot (y): %.2f %% (or %.4f m if zero-ref)\n", OS_py, OS_py);

fprintf("Settling Time (x): %.3f s\n", Ts_px);
fprintf("Settling Time (y): %.3f s\n", Ts_py);

fprintf("=====================================================\n");

