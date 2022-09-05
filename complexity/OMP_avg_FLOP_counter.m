function comp_avg = OMP_avg_FLOP_counter(N_samples, doa_true, doa_samples, SNRdB, M0, T, K_sparse)
% K_sparse is the maximum number of incoming signals
G = length(doa_samples); % Number of grids

% System model settings
c = 3e8; fc = 1e9; lamda = c/fc; d = 0.5*lamda;
M = floor(M0/2);
array_geom = [-M:1:M]'*d;
% LNA parameters    
Dy = 30*lamda; alpha = 2;

SNRdB = 10; N_coherent = 1;
var = 0.45; % Medium value of the uniform distribution for attenuation coefficient factor
N_signals = length(doa_true);
doa_true = doa_true + round(randn(1, N_signals), 1);

% Build a dictionany of atoms
Phi = []; 
for angle = doa_samples
    tmp = [-M:1:M]' - Dy*sin(angle*pi/180)/lamda;
    P_curr = sqrt(alpha)*sin(pi*tmp)./(pi*tmp);
%     phase_shift_array = 2*pi*array_geom/lamda*sin(angle*pi/180); 
%     P_curr = exp(-1j*phase_shift_array);
    Phi = [Phi, P_curr];
end

% Create samples and count FLOPs
comp_arr = zeros(1, N_samples);
for n = 1:N_samples
    
    array_signal = 0;
    A_arr = []; S = [];
    for sig = 1:N_signals
        tmp = [-M:1:M]' - Dy*sin(doa_true(sig)*pi/180)/lamda;
        A_curr = sqrt(alpha)*sin(pi*tmp)./(pi*tmp);
        if sig <= N_coherent
            if sig == 1
                S_0 = normrnd(0, 1, [1, T]) + 1j*normrnd(0, 1, [1, T]);
                S_curr = 1*S_0;
            else
                S_curr = attenuation_coef(var)*S_0;
            end
        else
            S_curr = normrnd(0, 1, [1, T]) + 1j*normrnd(0, 1, [1, T]);
        end
        S_curr = 10^(SNRdB/20)*S_curr/sqrt(2);
    %     array_signal = array_signal + A_ULA*S;
        A_arr = [A_arr, A_curr];
        S = [S; S_curr];
    end
    
    N = (normrnd(0, 1, [M0, T]) + 1j*normrnd(0, 1, [M0, T]))/sqrt(2);
    X = A_arr*S + N;

    % Count all Flops
    tol = 1e-6;
    for t = 1:T, 
        [~, N_it] = CS_OMP(X(:, t), Phi, K_sparse, tol); 
        % Flops of 1 example
        for k = 1:N_it
            comp_arr(n) = comp_arr(n) + (G - k + 1)*(2*M0 + 1) + 2*M0*k^2 + 2*M0*k;
        end
    end
    
end
comp_avg = mean(comp_arr);

end
