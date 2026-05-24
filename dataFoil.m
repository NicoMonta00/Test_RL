function Vela = dataFoil()


Vela.m = 4.3;                                       % mass [kg]
Vela.b = 2.06/2;                                      % wingspan [m]
Vela.c = 0.8;                                      % mean aero chord [m]
Vela.S = 1.64;                                         % parafoil surface [m^2]
Vela.I = [0.42, 0, 0.03; 
                0, 0.4, 0; 
                0.03, 0, 0.053];% inertia matrix parafoil+payload [3x3] [kg m^2]
%Vela.W= [0;0;0];                                    % wind vector [1x3] [m/s]
Vela.g0 = 9.81;                                     % gravity acceleration [m/s^2]

% Aerodynamics coefficients
% Vela.cd0         = 0.76;      % nostro
Vela.cd0 = 0.2284;              % loro
Vela.cd_alpha_2  = 0.0991;    % nostro
% Vela.cd_alpha_2 = 0.1;       % loro
% Vela.cl0         = 0.2947;    % nostro
Vela.cl0 = 0.1170;
% Vela.cl_alpha    = 2.02;      % nostro
Vela.cl_alpha = 0.8306;
% Vela.cm0         = 0.0820;    % nostro
Vela.cm0 = 0.2642; 
% Vela.cm_alpha    = -0.108;   % nostro 
Vela.cm_alpha = -0.7405;
% Vela.cm_q        = -0.1624;   % nostro
Vela.cm_q = -1.49;
% Vela.cl_p        = -0.1370;   % nostro
Vela.cl_p = -0.84;
% Vela.cL_delta_a  = 0.1138;    % nostro
Vela.cL_delta_a =  -0.002;
% Vela.cn_r        = -0.3044;   % nostro
Vela.cn_r = -0.27;
% Vela.cn_delta_a  = 0.0048;    % nostro
Vela.cn_delta_a = 0.0044;
Vela.cd_delta_s  = 0.0474;    % nostro
% Vela.cd_delta_s = 0.05;
Vela.delta_s_max = 0.0048;      % nostro
% Vela.delta_s_max = 0.1;
Vela.cl_phi      = -0.0100;   %tbd
Vela.cl_delta_a  = -0.0063;   % nostro
% Vela.cl_delta_a = 0.0010;     
%Vela.H0 = 400;                                       % starting altitude [m]

%% Coef aggiuntivi per pitch

Vela.cL_delta_e = 0.12;
Vela.cd_delta_e = 0.02;
Vela.cm_delta_e = -0.06;

end