clear all, clc, close all

env = 'cohe';
model_type = 'CFAR';
feature_input = 'diag_spec';
% SNRdB_arr = [-10:2.5:10];
SNRdB_arr = [0];
delta_SNRdB = 4;
N_snapshots_arr = [10:10:100];
% N_snapshots_arr = [40];
% N_antenas_arr = [65];
N_antenas = 65;

% Initialize CFAR detector
N_train = 8; N_guard = 2; fa_prob = 1e-1;
cfar = phased.CFARDetector('Method', 'CA', 'NumTrainingCells', N_train, 'NumGuardCells', N_guard, ...
                            'ThresholdOutputPort', true, 'ThresholdFactor', 'Auto',...
                            'ProbabilityFalseAlarm', fa_prob);

results = zeros(length(SNRdB_arr), length(N_snapshots_arr));
for idx_SNRdB =1:length(SNRdB_arr)
    SNRdB = SNRdB_arr(idx_SNRdB);
    for idx_snap = 1:length(N_snapshots_arr)
        N_snapshots = N_snapshots_arr(idx_snap);
        path = sprintf('./dataset_CFAR/%s/testset_Nsignals_%s_dB_%d_delta_%d_antenas_%d_snap.mat',...
                env, num2str(SNRdB), delta_SNRdB, N_antenas, N_snapshots);
        data = load(path);
        spectrum_arr = data.diag_spec; onehot_arr =  data.onehot;
        N_samples = length(spectrum_arr);
        sprintf('Dataset: SNRdB=%s, N_snapshots=%d', num2str(SNRdB), N_snapshots)
        for idx = progress(1:N_samples)
            spec = spectrum_arr(idx, :); onehot = onehot_arr(idx, :); N_signals_true = find(onehot)-1;
            spec_norm = normalize_spectrum(spec);
            [bin_peaks, thres] = cfar(spec_norm', 1:N_antenas);
            idx_peaks = find(bin_peaks);
            if idx_peaks > 0
                groups = indices_isolator(idx_peaks, 1);
                N_signals_pred = length(groups);
            else
                N_signals_pred = 0;
            end
            results(idx_SNRdB, idx_snap) = results(idx_SNRdB, idx_snap) + (N_signals_true==N_signals_pred);
        end
        % Averaging
        results(idx_SNRdB, idx_snap) = results(idx_SNRdB, idx_snap)/N_samples;
    end
end

if length(SNRdB_arr) > 1
    path = sprintf('./results/dec_prob_SNR_%s_%s_%s_%dsnaps.mat', env, model_type, feature_input, N_snapshots);
    save(path, 'results');
elseif length(N_snapshots_arr) > 1
    path = sprintf('./results/dec_prob_snapshots_%s_%s_%s_%sdB.mat', env, model_type, feature_input, num2str(SNRdB));
    save(path, 'results');
end
