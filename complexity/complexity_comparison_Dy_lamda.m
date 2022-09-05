% Complexity comparison of PSCNet (with different values of Dy/lamda) and ECNet
clear all; clc; close all
M0_arr = [53:2:113];
T = 40;
comp_proposed_full = zeros(1, length(M0_arr)); 
comp_proposed_cut_1 = zeros(1, length(M0_arr)); 
comp_proposed_cut_2 = zeros(1, length(M0_arr)); 
comp_ECNet = zeros(1, length(M0_arr)); 

% Parameters for CNN model
N_filters = 8; kernel_size = 3; N_conv = 5; input_shape_FC_CNN = N_filters*kernel_size; 

for m = 1:length(M0_arr)
    M0 = M0_arr(m); N_signals_max = round(M0/2);
    Dy_lamda_1 = 30; Dy_lamda_2 = 40; Dy_lamda_3 = 50
    M_1 = min(ceil(Dy_lamda_1), floor(M0/2)); M0_1 = 2*M_1 + 1;
    M_2 = min(ceil(Dy_lamda_2), floor(M0/2)); M0_2 = 2*M_2 + 1;
    M_3 = min(ceil(Dy_lamda_3), floor(M0/2)); M0_3 = 2*M_3 + 1;

    %% Proposed model
    % Parameters of full-input-model
    N_neurons_CNN = M0; N_FC_CNN = 2; 

    comp_FE = (2*T - 1)*M0;
    comp_conv = 2*M0*kernel_size*N_filters*(N_conv*N_filters + 1);
    [~, comp_FC_CNN] = distribute_neurons(N_filters*kernel_size, N_FC_CNN, N_neurons_CNN, N_signals_max);
    comp_proposed_full(m) = comp_FE + comp_conv + comp_FC_CNN;

    % Parameters of cut-off-model 1
    N_neurons_CNN = M0_1; N_FC_CNN = 2; 

    comp_FE = (2*T - 1)*M0_1;
    comp_conv = 2*M0_1*kernel_size*N_filters*(N_conv*N_filters + 1);
    [~, comp_FC_CNN] = distribute_neurons(N_filters*kernel_size, N_FC_CNN, N_neurons_CNN, N_signals_max);
    comp_proposed_cut_1(m) = comp_FE + comp_conv + comp_FC_CNN;

    % Parameters of cut-off-model 2
    N_neurons_CNN = M0_2; N_FC_CNN = 2; 

    comp_FE = (2*T - 1)*M0_2;
    comp_conv = 2*M0_2*kernel_size*N_filters*(N_conv*N_filters + 1);
    [~, comp_FC_CNN] = distribute_neurons(N_filters*kernel_size, N_FC_CNN, N_neurons_CNN, N_signals_max);
    comp_proposed_cut_2(m) = comp_FE + comp_conv + comp_FC_CNN;

    % Parameters of cut-off-model 3
    N_neurons_CNN = M0_3; N_FC_CNN = 2; 

    comp_FE = (2*T - 1)*M0_3;
    comp_conv = 2*M0_3*kernel_size*N_filters*(N_conv*N_filters + 1);
    [~, comp_FC_CNN] = distribute_neurons(N_filters*kernel_size, N_FC_CNN, N_neurons_CNN, N_signals_max);
    comp_proposed_cut_3(m) = comp_FE + comp_conv + comp_FC_CNN;

    %% ECNet
    % Parameters of ECNet
    N_neurons_DNN = 5*M0; N_FC_DNN = 4; 
    comp_conv_mat = (M0^2 + M0)*T/2;
    comp_EVD = 2*M0^3/3;
    [~, comp_FC_DNN] = distribute_neurons(M0, N_FC_DNN, N_neurons_DNN, N_signals_max);
    comp_ECNet(m) = comp_conv_mat + comp_EVD + comp_FC_DNN;
end

f = figure()
linewidth = 1.4;

plot(M0_arr, comp_proposed_full, 'linewidth', linewidth, 'linestyle', '-', 'color', [0, 0, 0]), hold on
plot(M0_arr, comp_proposed_cut_1, 'linewidth', linewidth, 'linestyle', '--', 'color', [1, 0.2, 0.2]), hold on
plot(M0_arr, comp_proposed_cut_2, 'linewidth', linewidth, 'linestyle', '--', 'color', [0, 0.4470, 0.7410]), hold on
plot(M0_arr, comp_proposed_cut_3, 'linewidth', linewidth, 'linestyle', '--', 'color', [0.9290 0.6940 0.1250]), hold on
plot(M0_arr, comp_ECNet, 'linewidth', linewidth, 'linestyle', '--', 'color', [0.4660 0.6740 0.1880]), hold on

annotation('rectangle', 'position', [0.725,0.2238,0.1304,0.0476])
annotation('textarrow', 'position', [0.8554,0.2714,0.0125,0.0738], 'linestyle', ':')
annotation('textarrow', 'position', [0.725,0.269047619047619,-0.123214285714286,0.076190476190476], 'linestyle', ':')

xlabel('Number of antennas $M_0$', 'interpreter', 'latex')
ylabel('Number of operations (FLOPs)', 'interpreter', 'latex')

lgd = legend({'PSCNet (${{\bf{r}}^{\left[ 0 \right]}} = {\bf{r}}$)', ...
        'PSCNet (${\raise0.7ex\hbox{${{D_y}}$} \!\mathord{\left/{\vphantom {{{D_y}} \lambda }}\right.\kern-\nulldelimiterspace}\!\lower0.7ex\hbox{$\lambda $}} = 30$, ${{\bf{r}}^{\left[ 0 \right]}} = \overline {\bf{r}} $)', ...
        'PSCNet (${\raise0.7ex\hbox{${{D_y}}$} \!\mathord{\left/{\vphantom {{{D_y}} \lambda }}\right.\kern-\nulldelimiterspace}\!\lower0.7ex\hbox{$\lambda $}} = 40$, ${{\bf{r}}^{\left[ 0 \right]}} = \overline {\bf{r}} $)', ...
        'PSCNet (${\raise0.7ex\hbox{${{D_y}}$} \!\mathord{\left/{\vphantom {{{D_y}} \lambda }}\right.\kern-\nulldelimiterspace}\!\lower0.7ex\hbox{$\lambda $}} = 50$, ${{\bf{r}}^{\left[ 0 \right]}} = \overline {\bf{r}} $)', ...
        'ECNet'}, 'Interpreter','latex', 'location', 'NW', 'FontSize', 12);
xlim([min(M0_arr), max(M0_arr)])
% hold off

axes('position', [0.6,0.3452380952381,0.267857142857141,0.109523809523806])
box on % put box around new pair of axes
idx = (97 <= M0_arr & M0_arr <= 109);
plot(M0_arr(idx), comp_proposed_full(idx), 'linewidth', linewidth, 'linestyle', '-', 'color', [0, 0, 0]), hold on
% plot(M0_arr(idx), comp_proposed_cut_1(idx), 'linewidth', linewidth, 'linestyle', '--', 'color', [0, 0.4470, 0.7410]), hold on
% plot(M0_arr(idx), comp_proposed_cut_2(idx), 'linewidth', linewidth, 'linestyle', '--', 'color', [0.9290 0.6940 0.1250]), hold on
plot(M0_arr(idx), comp_proposed_cut_3(idx), 'linewidth', linewidth, 'linestyle', '-.', 'color', [0.9290 0.6940 0.1250]), hold on
xlim([97, 109])
xticks([101])
% ylim([215000, 250000])
print -depsc2 complexity_comparison_transfer.eps




