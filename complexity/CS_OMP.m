function [x_k, N_it] = CS_OMP(b, A, K, tol)
% Assume b = A*x + noise, where A is a NxM dictionary, x is a K-sparse signal
[M, N] = size(A);
if nargin == 3, tol = 1e-6; end
% % Initialization 
r_k = b;
A_k = []; Lamda = [];
A_bar = A; % A_copy only contains the unexamined columns
hist_err = [];

% Loops
for k = 1:K
    Ar_k = A_bar'*r_k;
    [~, lamda_k] = max(abs(Ar_k));
    Lamda = [Lamda, lamda_k];
    A_k = [A_k, A(:, lamda_k)];
    x_k = zeros(N, 1);
    x_k(Lamda) = pinv(A_k)*b;
    b_k = A*x_k;
    r_k = b - b_k;
    % Update A_copy
    A_bar(:, lamda_k) = zeros(M, 1);
%     hist_err(end+1) = norm(x_k-x)/norm(x);
    hist_err(end+1) = norm(r_k);
    if hist_err(end) < tol, break, end
end
N_it = k;

end