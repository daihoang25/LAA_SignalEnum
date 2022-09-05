% Complexity comparison of all methods
clear all; clc; close all
M0_arr = [55:2:75];
T_arr = [1, 10:10:100];
comp_proposed = zeros(length(M0_arr), length(T_arr)); comp_ECNet = zeros(length(M0_arr), length(T_arr)); 
comp_AIC_MDL = zeros(length(M0_arr), length(T_arr)); comp_ER = zeros(length(M0_arr), length(T_arr)); 
comp_SORTE = zeros(length(M0_arr), length(T_arr)); 
% Parameters for CNN model
N_filters = 8; kernel_size = 3; N_conv = 5; input_shape_FC_CNN = N_filters*kernel_size; 

for m = 1:length(M0_arr)
    for t = 1:length(T_arr)
        M0 = M0_arr(m); T = T_arr(t); N_signals_max = round(M0/2);
        %% Proposed model
        % Parameters of FC layers
        N_neurons_CNN = M0; N_FC_CNN = 2; 

        comp_FE = (2*T - 1)*M0;
        comp_conv = 2*M0*kernel_size*N_filters*(N_conv*N_filters + 1);
        [~, comp_FC_CNN] = distribute_neurons(N_filters*kernel_size, N_FC_CNN, N_neurons_CNN, N_signals_max);
        comp_proposed(m, t) = comp_FE + comp_conv + comp_FC_CNN;
        
        %% CFAR
        % Parameters of FC layers
        N_tc = 8; % Number of training cells
        N_g = 2; % Number of guard cells
        N_s = 0.5*(N_tc + N_g); % Number of cells in each side
        comp_FE = (2*T - 1)*M0;
        comp_CA = (M0 - 2*N_s)*(2*N_s + N_g); % Cell averaging
        comp_CFAR(m, t) = comp_FE + comp_CA;
        
        %% ECNet
        % Parameters of ECNet
        N_neurons_DNN = 5*M0; N_FC_DNN = 4; 
        comp_conv_mat = (M0^2 + M0)*T/2;
        comp_EVD = 2*M0^3/3;
        [~, comp_FC_DNN] = distribute_neurons(M0, N_FC_DNN, N_neurons_DNN, N_signals_max);
        comp_ECNet(m, t) = comp_conv_mat + comp_EVD + comp_FC_DNN;
        
        %% AIC - MDL
        comp_conv_mat = (M0^2 + M0)*T/2;
        comp_EVD = 2*M0^3/3;
        comp_AIC_MDL_enum = M0^2;
        comp_AIC_MDL(m, t) = comp_conv_mat + comp_EVD + comp_AIC_MDL_enum;

        %% ER
        comp_conv_mat = (M0^2 + M0)*T/2;
        comp_EVD = 2*M0^3/3;
        comp_ER_enum = M0;
        comp_ER(m, t) = comp_conv_mat + comp_EVD + comp_ER_enum;
        
        %% SORTE
        comp_conv_mat = (M0^2 + M0)*T/2;
        comp_EVD = 2*M0^3/3;
        comp_SORTE_enum = 4*M0*(M0-1) - 2*(M0-2)*(M0-1) - (M0-1);
        comp_SORTE(m, t) = comp_conv_mat + comp_EVD + comp_SORTE_enum;
    end
end

f = figure()
linewidth = 1.4;

h=gca
% set(h, 'cameraposition', [-68.73934120892034,500.4046529921955,1989368.773458484])
s1 = surf(M0_arr, T_arr, comp_ECNet,'FaceAlpha',0.7); hold on
s1.EdgeColor = [0.4470, 0, 0.7410]; % interp, flat, none, color
s1.LineStyle = '--';
s1.LineWidth = linewidth;
s1.FaceColor = [0.4470, 0, 0.7410]; % interp, flat, none, color
s1.FaceAlpha = 0.5;

s2 = surf(M0_arr, T_arr, comp_SORTE,'FaceAlpha',0.7); hold on
s2.EdgeColor = [1 0.4940 1]; % interp, flat, none, color
s2.LineStyle = '-.';
s2.LineWidth = linewidth;
s2.FaceColor = [1 0.4940 1]; % interp, flat, none, color
s2.FaceAlpha = 0.5;

s3 = surf(M0_arr, T_arr, comp_AIC_MDL,'FaceAlpha',0.7); hold on
s3.EdgeColor = [0.4660 0.6740 0.1880]; % interp, flat, none, color
s3.LineStyle = '-.';
s3.LineWidth = linewidth;
s3.FaceColor = [0.4660 0.6740 0.1880]; % interp, flat, none, color
s3.FaceAlpha = 0.5;

s4 = surf(M0_arr, T_arr, comp_ER,'FaceAlpha',0.7); hold on
s4.EdgeColor = [0.9290 0.6940 0.1250]; % interp, flat, none, color
s4.LineStyle = '-.';
s4.LineWidth = linewidth;
s4.FaceColor = [0.9290 0.6940 0.1250]; % interp, flat, none, color
s4.FaceAlpha = 0.5;

s5 = surf(M0_arr, T_arr, comp_CFAR,'FaceAlpha',0.7); hold on
s5.EdgeColor = [0.2, 0.4, 1];
s5.LineStyle = '--';
s5.LineWidth = linewidth;
s5.FaceColor = [0.2, 0.4, 1];
s5.FaceAlpha = 0.5;

s6 = surf(M0_arr, T_arr, comp_proposed,'FaceAlpha',0.7); hold on
s6.EdgeColor = [1, 0, 0];
s6.LineStyle = '--';
s6.LineWidth = linewidth;
s6.FaceColor = [1, 0, 0];
s6.FaceAlpha = 0.5;

lgd = legend({'ECNet', 'SORTE', 'AIC - MDL', 'ER', 'CFAR', 'PSCNet'}, 'Interpreter','latex', ...
    'Position', [0.702018986816988,0.542188097622281,0.201190597794592,0.221190471422105]);
% lgd.FontSize = 10;
hold off
% xticks(M0_arr)
% title('Computational complexity')
axis([min(M0_arr), max(M0_arr), min(T_arr), max(T_arr)])
% xticks([min(N_antenas_arr):10:max(N_antenas_arr)]);
zlim([min(min([comp_CFAR, comp_ECNet])), max(max([comp_CFAR, comp_ECNet]))])
% xlabel('Number of antennas - $M_0$', 'Interpreter','latex')
% ylabel('Number of snapshots - $T$', 'Interpreter','latex')
xlabel('$M_0$', 'Interpreter','latex')
ylabel('$T$', 'Interpreter','latex')
zlabel('Number of operations (FLOPs)', 'Interpreter','latex')
% colorbar
% grid on
campos([-69.74489765217044,498.703427903138,2225782.123998445])
print -depsc2 complexity_comparison.eps

