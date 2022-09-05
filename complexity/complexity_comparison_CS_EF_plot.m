% Compare complexity between the spectrum formation based on
% energy-focusing property and compressive sensing

clear all, clc, close all

M0_arr = [53:2:113]; 
SNRdB = 10;
res_arr = [1, 0.5, 0.1]; 
doa_min = -60; doa_max = 60;
comp_EF_arr = zeros(1, length(M0_arr)); 
comp_CS_upper_arr = zeros(length(res_arr), length(M0_arr)); 
comp_CS_simu_arr = zeros(length(res_arr), length(M0_arr));
N_signals_max = 8;
N_samples = 100;
doa_true = [-58, -22, 56, 30, 15, -42];
T1 = 40; 
T2 = 10;

%% semilogy conigurations
figure()
linewidth = 1.5;
semilogy_space = 5;
marker_idx = 1:semilogy_space:length(M0_arr);

% T = 40 snapshots
T = T1; 
file_name_mat = strcat('complexity_comparison_CS_EF_', num2str(T), '_snaps.mat');
load(file_name_mat)

color = [0, 0, 0]; marker = '>';
semilogy(M0_arr, comp_EF_arr, 'linewidth', linewidth, 'linestyle', '-', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
% Resolution = 1
color = [1, 0, 0];
semilogy(M0_arr, comp_CS_upper_arr(1, :), 'linewidth', linewidth, 'linestyle', '-', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
semilogy(M0_arr, comp_CS_simu_arr(1, :), 'linewidth', linewidth, 'linestyle', '--', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
% Resolution = 0.5
color = [0, 0.6, 0];
semilogy(M0_arr, comp_CS_upper_arr(2, :), 'linewidth', linewidth, 'linestyle', '-', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
semilogy(M0_arr, comp_CS_simu_arr(2, :), 'linewidth', linewidth, 'linestyle', '--', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
% Resolution = 0.1
color = [0.2, 0.4, 1];
semilogy(M0_arr, comp_CS_upper_arr(3, :), 'linewidth', linewidth, 'linestyle', '-', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
semilogy(M0_arr, comp_CS_simu_arr(3, :), 'linewidth', linewidth, 'linestyle', '--', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on

% T = 10 snapshots
T = T2; 
file_name_mat = strcat('complexity_comparison_CS_EF_', num2str(T), '_snaps.mat');
load(file_name_mat)

color = [0, 0, 0]; marker = '<';
semilogy(M0_arr, comp_EF_arr, 'linewidth', linewidth, 'linestyle', '-', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
% Resolution = 1
color = [1, 0, 0];
semilogy(M0_arr, comp_CS_upper_arr(1, :), 'linewidth', linewidth, 'linestyle', '-', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
semilogy(M0_arr, comp_CS_simu_arr(1, :), 'linewidth', linewidth, 'linestyle', '--', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
% Resolution = 0.5
color = [0, 0.6, 0];
semilogy(M0_arr, comp_CS_upper_arr(2, :), 'linewidth', linewidth, 'linestyle', '-', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
semilogy(M0_arr, comp_CS_simu_arr(2, :), 'linewidth', linewidth, 'linestyle', '--', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
% Resolution = 0.1
color = [0.2, 0.4, 1];
semilogy(M0_arr, comp_CS_upper_arr(3, :), 'linewidth', linewidth, 'linestyle', '-', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on
semilogy(M0_arr, comp_CS_simu_arr(3, :), 'linewidth', linewidth, 'linestyle', '--', 'color', color, 'marker', marker, 'MarkerIndices', marker_idx), hold on

% Decorations
xlim([min(M0_arr), max(M0_arr)])
ylim([0, 10^(8.5)])
xlabel('Number of antennas $M_0$', 'interpreter', 'latex')
ylabel('Number of operations (FLOPs)', 'interpreter', 'latex')

s1 = scatter(0, 0, 'linewidth', linewidth, 'markeredgecolor', [0, 0, 0], 'marker', '>'), hold on
s2 = scatter(0, 0, 'linewidth', linewidth, 'markeredgecolor', [0, 0, 0], 'marker', '<'), hold on

a1 = semilogy(0, 0, 'linewidth', linewidth, 'linestyle', '-', 'color', [0.2, 0.4, 1]), hold on
a2 = semilogy(0, 0, 'linewidth', linewidth, 'linestyle', '--', 'color', [0.2, 0.4, 1]), hold on

b1 = semilogy(0, 0, 'linewidth', linewidth, 'linestyle', '-', 'color', [0, 0.6, 0]), hold on
b2 = semilogy(0, 0, 'linewidth', linewidth, 'linestyle', '--', 'color', [0, 0.6, 0]), hold on

c1 = semilogy(0, 0, 'linewidth', linewidth, 'linestyle', '-', 'color', [1, 0, 0]), hold on
c2 = semilogy(0, 0, 'linewidth', linewidth, 'linestyle', '--', 'color', [1, 0, 0]), hold on

d = semilogy(0, 0, 'linewidth', linewidth, 'linestyle', '-', 'color', [0, 0, 0]), hold off

legend([s1, s2, a1, a2, b1, b2, c1, c2, d], ["$T = 40$ snapshots", "$T = 10$ snapshots", ...
    "OMP-Upper bound - $\rho  = 0.1^\circ$", "OMP-Simulation - $\rho  = 0.1^\circ$", ...
    "OMP-Upper bound - $\rho  = 0.5^\circ$", "OMP-Simulation - $\rho  = 0.5^\circ$", ...
    "OMP-Upper bound - $\rho  = 1^\circ$", "OMP-Simulation - $\rho  = 1^\circ$", ...
    "Energy-focusing $\left( {{\bf{r}}^{\left[ 0 \right]}} = {\bf{r}} \right)$"], 'interpreter', 'latex', 'position', [0.144129185626217,0.237698413413669,0.370602953618251,0.32761904046649])

file_name_eps = strcat('complexity_comparison_CS_EF_', num2str(T1), '_vs_', num2str(T2), '_snaps.eps');
print(file_name_eps, '-depsc2')

