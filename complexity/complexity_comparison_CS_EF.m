% Compare complexity between the spectrum formation based on
% energy-focusing property and compressive sensing

clear all, clc, close all

M0_arr = [53:2:113]; 
T = 40; SNRdB = 10;
res_arr = [1, 0.5, 0.1]; 
doa_min = -60; doa_max = 60;
comp_EF_arr = zeros(1, length(M0_arr)); 
comp_CS_upper_arr = zeros(length(res_arr), length(M0_arr)); 
comp_CS_simu_arr = zeros(length(res_arr), length(M0_arr));
N_signals_max = 8;
N_samples = 100;
doa_true = [-58, -22, 56, 30, 15, -42];
file_name_mat = strcat('complexity_comparison_CS_EF_', num2str(T), '_snaps.mat');
file_name_eps = strcat('complexity_comparison_CS_EF_', num2str(T), '_snaps.eps');

run_from_scratch = false;

if run_from_scratch
    for m = 1:length(M0_arr)
        disp('---------------------')
        M0 = M0_arr(m); 
        Dy_lamda = 30;
        M = floor(M0/2); M0 = 2*M + 1;
    
        %% Energy-focusing based formation
        comp_EF = (2*T - 1)*M0;
        comp_EF_arr(m) = comp_EF;
    
        for r = 1:length(res_arr)
            % Compute the number of grids
            res = res_arr(r);
            doa_samples = doa_min:res:(doa_max-res);
            G = length(doa_samples); % Number of grids
    
            disp(strcat('Resolution=', num2str(res), ', M0=', num2str(M0)))
    
            %% Compressive sensing based formation (Upper bound)
            comp_CS = 0;
            % Compressive sensing steps
            for k = 1:N_signals_max
                comp_CS = comp_CS + (G - k + 1)*(2*M0 + 1) + 2*M0*k^2 + 2*M0*k;
            end
            % For T snapshots, the CS iterations have to be repeated T times
            comp_CS = comp_CS*T;
            % Covariance matrix multiplication
            comp_CS = comp_CS + G^2*T + G*T - G^2/2 - G/2;
            comp_CS_upper_arr(r, m) = comp_CS;
        
            %% Compressive sensing based formation (Simulation)
            comp_CS_simu_arr(r, m) = OMP_avg_FLOP_counter(N_samples, doa_true, doa_samples, SNRdB, M0, T, N_signals_max);
        end
    end
    save(file_name_mat, 'comp_EF_arr', 'comp_CS_upper_arr', 'comp_CS_simu_arr')

else % Load results then plot the figure
    load(file_name_mat)
end

figure()
linewidth = 1.4;

plot(M0_arr, comp_EF_arr, 'linewidth', 1, 'linestyle', '-', 'color', [0, 0, 0], ...
    'marker', 'o', 'markerfacecolor', [0, 0, 0]), hold on
% Resolution = 1
c = [1, 0, 0];
plot(M0_arr, comp_CS_upper_arr(1, :), 'linewidth', linewidth, 'linestyle', '--', 'color', c), hold on
plot(M0_arr, comp_CS_simu_arr(1, :), 'linewidth', linewidth, 'linestyle', '-', 'color', c), hold on
% Resolution = 0.5
c = [0, 0.6, 0];
plot(M0_arr, comp_CS_upper_arr(2, :), 'linewidth', linewidth, 'linestyle', '--', 'color', c), hold on
plot(M0_arr, comp_CS_simu_arr(2, :), 'linewidth', linewidth, 'linestyle', '-', 'color', c), hold on
% Resolution = 0.1
c = [0.2, 0.4, 1];
plot(M0_arr, comp_CS_upper_arr(3, :), 'linewidth', linewidth, 'linestyle', '--', 'color', c), hold on
plot(M0_arr, comp_CS_simu_arr(3, :), 'linewidth', linewidth, 'linestyle', '-', ...
    'color', c), hold on

xlim([min(M0_arr), max(M0_arr)])
xlabel('Number of antennas $M_0$', 'interpreter', 'latex')
ylabel('Number of operations (FLOPs)', 'interpreter', 'latex')
legend('Energy-focusing (EF)', 'CS-Upper bound - $\rho  = 1^\circ$', 'CS-Simulation - $\rho  = 1^\circ$', ...
    'CS-Upper bound - $\rho  = 0.5^\circ$', 'CS-Simulation - $\rho  = 0.5^\circ$', ...
    'CS-Upper bound - $\rho  = 0.1^\circ$', 'CS-Simulation - $\rho  = 0.1^\circ$', ...
    'interpreter', 'latex', 'position', [0.528623229632591,0.247460311738272,0.349472008462647,0.256666672388712])

print(file_name_eps, '-depsc2')

