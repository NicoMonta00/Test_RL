function [curve_pts, cp_real, w_coeffs, angle_chord, dist_chord, L_out, T_out, t1_out] = ph7_teta_vincolato_new(P0, P1, t0, t1_min_rad, t1_max_rad, k0_target, k1_target, L_real, varargin)
%% SOLVE_PH7_L_OPT_WITH_VARIABLE_TANGENT (CON VINCOLO ANGOLO FINALE E SLACK VARIABLE)
% Versione OTTIMIZZATA PER SIMULINK
% Restituisce scalari puri invece di struct per compatibilità coder.extrinsic
%
% INPUT:
%   P0, P1       - Punti endpoint [x, y]
%   t0           - Angolo tangente iniziale (FISSO, radianti)
%   t1_min_rad   - Range minimo angolo finale (rad)
%   t1_max_rad   - Range massimo angolo finale (rad)
%   k0_target    - Curvatura iniziale
%   k1_target    - Curvatura finale
%   L_real       - Lunghezza arco target
%   varargin     - Flags: 'verbose', 'diagnostics', 'use_exact'

    % ============================================================
    % PARSING INPUT E FLAGS
    % ============================================================
    p = inputParser;
    addParameter(p, 'diagnostics', false, @islogical);
    addParameter(p, 'verbose', false, @islogical);
    addParameter(p, 'use_exact', true, @islogical); 
    parse(p, varargin{:});
    
    use_diagnostics = p.Results.diagnostics;
    use_verbose = p.Results.verbose;
    use_exact = p.Results.use_exact;
    
    % ============================================================
    % GEOMETRIA E CANONICALIZZAZIONE
    % ============================================================
    vec_diff = P1 - P0;
    dist_chord = norm(vec_diff);
    if dist_chord < 1e-3, dist_chord = 1e-3; end
    angle_chord = atan2(vec_diff(2), vec_diff(1));
    
    % Normalizzazione
    L_canon = L_real / dist_chord;
    k0_canon = k0_target * dist_chord;
    k1_canon = k1_target * dist_chord;
    
    % Angolo iniziale e limiti relativi alla corda
    t0_rel = t0 - angle_chord;
    t1_min_rel = t1_min_rad - angle_chord;
    t1_max_rel = t1_max_rad - angle_chord;
    
    if use_verbose
        fprintf('\n%s\n', repmat('=', 1, 70));
        fprintf('SETUP OTTIMIZZAZIONE PH7 (SLACK VARIABLE ARCHITECTURE)\n');
        fprintf('%s\n', repmat('=', 1, 70));
    end
    
    % ============================================================
    % GENERAZIONE DINAMICA DEL GUESS INIZIALE A 7 VARIABILI
    % ============================================================
    t1_rel_guess = (t1_min_rel + t1_max_rel) / 2;
    d_guess = max(sqrt(L_canon), 0.1); 
    
    w0_guess = d_guess * exp(1i * t0_rel / 2);
    w3_guess = d_guess * exp(1i * t1_rel_guess / 2);
    w1_guess = (2/3)*w0_guess + (1/3)*w3_guess;
    w2_guess = (1/3)*w0_guess + (2/3)*w3_guess;
    
    u1_g = real(w1_guess); v1_g = imag(w1_guess);
    u2_g = real(w2_guess); v2_g = imag(w2_guess);
    
    % VETTORE X0 ESPANSO A 7 ELEMENTI (Epsilon finale inizializzato a 0.0)
    x0 = [u1_g, u2_g, v1_g, v2_g, d_guess, t1_rel_guess, 0.0];
    
    % ============================================================
    % SETUP SOLUTORE
    % ============================================================
    options = optimoptions('fmincon', ...
        'Display', 'off', ...
        'Algorithm', 'interior-point', ...
        'MaxFunctionEvaluations', 5000, ...
        'StepTolerance', 1e-8, ...       % Tolleranza allargata per la stabilità
        'OptimalityTolerance', 1e-3, ...
        'ConstraintTolerance', 1e-4);

    options = optimoptions('fmincon', ...
        'Display', 'off', ...
        'Algorithm', 'sqp', ...          
        'MaxFunctionEvaluations', 5000, ...
        'StepTolerance', 1e-8, ...
        'OptimalityTolerance', 1e-3, ...
        'ConstraintTolerance', 1e-3);
        
    % CLOSURES AGGIORNATE:
    % Obiettivo: Non prende più L_canon, usa solo t0_rel e x
    fun = @(x) objective_energy_var_tangent(x, t0_rel, dist_chord, L_canon);
    
    % Vincoli: Prende L_canon e dist_chord per calcolare l'errore epsilon e il raggio reale
    nonlcon = @(x) constraints_geometric_var_tangent(x, t0_rel, L_canon, dist_chord);
    
    % BOUNDS ESPANSI (Epsilon libero da -Inf a +Inf)
    lb = [-Inf, -Inf, -Inf, -Inf, 0.1, t1_min_rel, -Inf];
    ub = [Inf, Inf, Inf, Inf, 100, t1_max_rel,  Inf];
    
    % ============================================================
    % ESECUZIONE FMINCON
    % ============================================================
    tic;
    [x_sol, ~, exitflag, output] = fmincon(fun, x0, [], [], [], [], lb, ub, nonlcon, options);
    t_solver = toc; 
    
    if exitflag < 0
        warning('Solver non ha convergeto. exitflag = %d. Uso guess iniziale.', exitflag);
        x_sol = x0; 
    end
    
    if use_verbose
        fprintf(' └─ Exitflag: %d (Epsilon usato: %.4f m reali)\n', exitflag, x_sol(7)*dist_chord);
    end

    % ============================================================
    % POST-PROCESSING: RICOSTRUZIONE DELLA CURVA
    % ============================================================
    tic; 
    
    % ESTRAZIONE (Solo i primi 6 elementi, epsilon è x_sol(7) e viene ignorato qui)
    u1_opt = x_sol(1); u2_opt = x_sol(2);
    v1_opt = x_sol(3); v2_opt = x_sol(4);
    d_opt = x_sol(5); t1_rel_opt = x_sol(6);
    
    w0_opt = d_opt * exp(1i * t0_rel / 2);
    w3_opt = d_opt * exp(1i * t1_rel_opt / 2);
    w1_opt = u1_opt + 1i * v1_opt;
    w2_opt = u2_opt + 1i * v2_opt;
    
    % Calcolo Lunghezza Reale Ottenuta (Per sicurezza/log)
    gp = [-0.93247, -0.66121, -0.23862, 0.23862, 0.66121, 0.93247];
    gw = [0.17132, 0.36076, 0.46791, 0.46791, 0.36076, 0.17132];
    L_ottenuto_canon = 0;
    for k=1:6
        t = 0.5 * gp(k) + 0.5;
        b0=(1-t)^3; b1=3*(1-t)^2*t; b2=3*(1-t)*t^2; b3=t^3;
        w_val = w0_opt*b0 + w1_opt*b1 + w2_opt*b2 + w3_opt*b3;
        L_ottenuto_canon = L_ottenuto_canon + gw(k) * abs(w_val)^2;
    end
    L_ottenuto_canon = (L_ottenuto_canon * 0.5);
    L_ottenuto_reale = L_ottenuto_canon * dist_chord;
    errore_relativo = (L_ottenuto_reale - L_real) / L_real * 100;
    
    % Generazione Curva
    w_coeffs = [w0_opt, w1_opt, w2_opt, w3_opt];
    t_vals = linspace(0, 1, 200); 
    
    if use_exact
        cp_canon = compute_ph7_exact_cp(w_coeffs);
        evaluate_func = @evaluate_bezier_exact;
    else
        cp_canon = compute_control_points_old(w_coeffs);
        evaluate_func = @evaluate_bezier_old;
    end
    
    rot_factor = exp(1i * angle_chord);
    P0_complex = P0(1) + 1i*P0(2);
    cp_real = P0_complex + cp_canon .* dist_chord .* rot_factor;
    curve_pts = evaluate_func(cp_real, t_vals);
    
    t_postproc = toc; 
    
    % Output
    L_out  = L_ottenuto_reale;
    T_out  = t_solver + t_postproc;
    t1_out = t1_rel_opt + angle_chord;
    
    % Diagnostica
    if use_diagnostics
        figure(10); clf; axis equal; hold on;
        plot(real(curve_pts), imag(curve_pts), 'b-', 'LineWidth', 2);
        if use_exact
            plot(real(cp_real), imag(cp_real), 'c--.', 'MarkerSize', 8);
        end
        plot(P0(1), P0(2), 'go'); plot(P1(1), P1(2), 'ro');
        title(sprintf('PH7 Err: %.3f%% | Epsilon USATO: %.3fm', errore_relativo, x_sol(7)*dist_chord));
        grid on; drawnow;
    end
end

function E = objective_energy_var_tangent(x, t0_rel, dist_chord, L_canon)
    % Estrazione 7 variabili (Compresa la slack variable)
    u1=x(1); u2=x(2); v1=x(3); v2=x(4); d=x(5); t1_rel=x(6);
    epsilon = x(7); 
    
    w_vec = [d * exp(1i * t0_rel / 2); u1 + 1i * v1; u2 + 1i * v2; d * exp(1i * t1_rel / 2)];
    
    persistent B_gauss B_gauss_der 
    if isempty(B_gauss)
        gp = [-0.93247, -0.66121, -0.23862, 0.23862, 0.66121, 0.93247];
        t = 0.5 * gp + 0.5;
        B_gauss = [(1-t).^3; 3.*(1-t).^2.*t; 3.*(1-t).*t.^2; t.^3].'; 
        B_gauss_der = [-3.*(1-t).^2;  3.*(1-t).^2 - 6.*t.*(1-t);  6.*t.*(1-t) - 3.*t.^2;  3.*t.^2].';
    end
    
    w_eval = B_gauss * w_vec; 
    w_der_eval = B_gauss_der * w_vec; 
    
    % 1. Bending Energy
    Bending_Energy = sum(abs(diff(w_vec)).^2); 
    
    % 2. Pseudo-Jerk (Senza denominatore)
    jerk_penalty = sum(abs(diff(w_der_eval)).^2);
    
    % 3. MURO DI GOMMA POLINOMIALE (Il trucco anti-singolarità)
    R_min_reale = 20.0; % Metri
    k_max_reale = 1.0 / R_min_reale;
    k_max_canon = k_max_reale * dist_chord; 
    
    % Separiamo Numeratore e Denominatore (N e D)
    % Usiamo abs() sul numeratore perché il veicolo potrebbe curvare a destra o a sinistra
    Num_k = abs(2 * imag(conj(w_eval) .* w_der_eval)); 
    Den_k = abs(w_eval).^4;
    k = Num_k./(Den_k + 1e-6);
    k_squared = sum(k.^2);

    % k solo numeratore per stabilità
    k_num = sum(Num_k.^2);

    V_canon = abs(w_eval).^2;
    V_min_sicura = 1;
    deficit_velocita = max(0, V_min_sicura - V_canon);
    anti_stall_penalty = sum(deficit_velocita.^2);


    V_media_ideale = L_canon;
    differenza_velocita = V_canon - V_media_ideale;
    elastico_penalty = sum(differenza_velocita.^2);

    % Nessuna divisione = Nessuna singolarità possibile!
    eccesso_polinomiale = max(0, Num_k - k_max_canon .* Den_k);
    soft_wall_penalty = sum(eccesso_polinomiale.^2);
    
    % --- PESI BILANCIATI  ---
    peso_bending    = 1;
    peso_uniformita = 0;    
    peso_curvatura = 50;
    peso_soft_wall  = 0;
    peso_slack      = 1e5;

    peso_bending    = 1;
    peso_uniformita = 10;    
    peso_curvatura = 50;
    peso_soft_wall  = 1000; %R = 40
    peso_slack      = 1e5;

    peso_bending    = 1;
    peso_uniformita = 0;    
    peso_curvatura = 0;
    peso_soft_wall  = 100000; % R = 40
    peso_slack      = 1e5;
    peso_k_num = 100;
    peso_stallo = 5000; % V = 0.5

    peso_bending    = 1;
    peso_uniformita = 0;    
    peso_curvatura = 0;
    peso_soft_wall  = 0;
    peso_slack      = 1e6;
    peso_k_num = 0;
    peso_stallo = 1e4; % V = 1

    peso_bending    = 1;
    peso_uniformita = 0;    
    peso_curvatura = 0;
    peso_soft_wall  = 0;
    peso_slack      = 1e5;
    peso_k_num = 0;
    peso_stallo = 1e4; % V = 1
    peso_elastico = 0;

    peso_bending    = 1;
    peso_uniformita = 0;    
    peso_curvatura = 0;
    peso_soft_wall  = 0;
    peso_slack      = 0;
    peso_k_num = 0;
    peso_stallo = 0;
    peso_elastico = 50;

    % Costo Totale
    E = peso_bending * Bending_Energy + ...
        peso_uniformita * jerk_penalty + ...
        peso_slack * (epsilon^2) + ...
        peso_soft_wall * soft_wall_penalty + ...
        peso_curvatura * k_squared + ...
        peso_k_num * k_num + ...
        peso_stallo * anti_stall_penalty + ...
        peso_elastico * elastico_penalty;
end

function [c, ceq] = constraints_geometric_var_tangent(x, t0_rel, L_target, d_chord)
    % Estrazione 7 variabili
    u1=x(1); u2=x(2); v1=x(3); v2=x(4); d=x(5); t1_rel=x(6);
    epsilon = x(7); 
    
    w_vec = [d * exp(1i * t0_rel / 2); u1 + 1i * v1; u2 + 1i * v2; d * exp(1i * t1_rel / 2)];
    
    persistent B_gauss  gw
    if isempty(B_gauss)
        gp = [-0.93247, -0.66121, -0.23862, 0.23862, 0.66121, 0.93247];
        gw = [0.17132, 0.36076, 0.46791, 0.46791, 0.36076, 0.17132] * 0.5;
        t = 0.5 * gp + 0.5;
        B_gauss = [(1-t).^3; 3.*(1-t).^2.*t; 3.*(1-t).*t.^2; t.^3].'; 
    end
    
    w_eval = B_gauss * w_vec; 
    
    % Calcoli integrali
    Delta_Complex = gw * (w_eval.^2);
    Length_Calc = gw * (abs(w_eval).^2); 
    
    % --- HARD CONSTRAINTS (Uguaglianze) ---
    % Imponiamo il target spaziale (X,Y) e il target di Lunghezza + epsilon
    ceq = [real(Delta_Complex) - 1.0; 
           imag(Delta_Complex) - 0.0;
           Length_Calc - L_target - epsilon];
           
    c = []; 
end

% ============================================================
% 1. FUNZIONI VECCHIE (LEGACY)
% ============================================================
function p_coeffs = compute_control_points_old(w)
    tt = linspace(0, 1, 500);
    ww = (w(1).*(1-tt).^3 + w(2).*3.*(1-tt).^2.*tt + w(3).*3.*(1-tt).*tt.^2 + w(4).*tt.^3).^2;
    p_coeffs = cumtrapz(tt, ww);
end
function pts = evaluate_bezier_old(cp_start, t_vals)
    tt_base = linspace(0, 1, length(cp_start));
    if isrow(cp_start), cp_start = cp_start.'; end
    pts = interp1(tt_base, cp_start, t_vals, 'spline');
    if iscolumn(pts), pts = pts.'; end
end
% ============================================================
% 2. FUNZIONI NUOVE (ESATTE & TURBO OTTIMIZZATE)
% ============================================================
function cp_canon = compute_ph7_exact_cp(w)
    w = w(:).'; 
    bin_3 = [1, 3, 3, 1];
    bin_6 = [1, 6, 15, 20, 15, 6, 1];
    w_scaled = w .* bin_3;
    q_scaled = conv(w_scaled, w_scaled);
    q = q_scaled ./ bin_6;
    cp_canon = cumsum([0, q / 7]);
end
function pts = evaluate_bezier_exact(cp, t_vals)
    n = length(cp) - 1;
    t = t_vals(:);
    cp = cp(:);
    persistent C_cached;
    if isempty(C_cached) || length(C_cached) ~= (n+1)
        C_cached = diag(rot90(pascal(n+1)))';
    end
    C = C_cached;
    rng = 0:n;
    T_pow = t .^ rng;
    OneMinusT_pow = (1 - t) .^ (n - rng);
    B = C .* T_pow .* OneMinusT_pow;
    pts = (B * cp).';
end
% ============================================================
% UTILS
% ============================================================
function result = iif(condition, true_val, false_val)
    if condition, result = true_val; else, result = false_val; end
end