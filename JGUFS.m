function [W,obj] = JGUFS(X,A,c,alpha,beta,gamma)

% input:
%   X -- data collection matrix, size n-by-d
%   A -- the initial graph affinity matrix
%   c -- the number of clusters
% output:
%   W -- the projection matrix, size d-by-c

lambda = 1e7;
[n,d] = size(X);

% initialize 'F' based on spectral clustering
D = diag(sum(A));
L = D - A;
[V,~] = eig(L);
F = V(:,1:c);
% F = rand(n,c);

% initialize 'M' as identity matrix
M = eye(d);

% precomputing for acceleration
XTX = X'*X;

iter = 1;
while iter < 50
    
    % update 'W'
    tmp = XTX + gamma * M;
    W = tmp \(X'*F);
            
    % update 'F'
    L = diag(sum(A)) - A;
    R = alpha * L + beta * (eye(n) - 2 * X *(tmp\X'));
    F = (lambda * F .* F)./(R*F + lambda*(F*F')*F + eps);
    F = F*diag(sqrt(1./(diag(F'*F)+eps)));
    clear tmp
    
    % update 'S'
    dist = L2_distance_1(F',F');
    S = zeros(n,n);
    for i = 1:n
        ai = A(i,:);
        di = dist(i,:);
        ad = ai - 0.5*alpha*di;
        S(i,:) = EProjSimplex_new(ad);
    end
    A = S;
    A = (A+A')/2;
    
    % update 'M'
    Wi = sqrt(sum(W.*W,2)+eps);
    M = diag(0.5./Wi);
    
    % objective value
    obj(iter) = norm(S-A,'fro')^2 + alpha*trace(F'*L*F) + ...
        beta*(norm(X*W-F,'fro')^2 + sum(sqrt(sum(W.*W,2))));
    
    % update 'iter'
    iter = iter + 1;
end

save F F