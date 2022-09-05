%% Neuron distributing function
function [N_hidden_arr, complexity] = distribute_neurons(input_shape, N_FC, N_neurons, N_signals_max)
    N_hidden_arr = []; complexity = 0;
    N_neurons_each = floor(N_neurons/N_FC);
    for l = 1:N_FC
        if l == 1
            N_hidden_arr(end+1) = N_neurons - (N_FC - 1)*N_neurons_each;
            complexity = complexity + input_shape*N_hidden_arr(end);
        else
            N_hidden_arr(end+1) = N_neurons_each;
            complexity = complexity + 2*N_hidden_arr(end-1)*N_hidden_arr(end);
        end
    end
    complexity = complexity + 2*N_hidden_arr(end)*N_signals_max; % Complexity at the output layer
end