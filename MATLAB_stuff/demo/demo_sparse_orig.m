%% Generate Data Set
clear; 
% dimension, np, nq
m = 500, np = 1500, nq = 1500
rng(10); adj = rand(m)<.001; adj = triu(adj, 1) + triu(adj, 1)'; thetaP = double(adj)*.45 + eye(m)*2;
rng(10); adj = rand(m)<.001-3e-5; adj = triu(adj, 1) + triu(adj, 1)'; thetaQ = double(adj)*.45 + eye(m)*2;
rng(1);
xp = mvnrnd(zeros(m,1),inv(thetaP),np)'; xq = mvnrnd(zeros(m,1),inv(thetaQ),nq)';
kp = kernel_linear(xp); kq = kernel_linear(xq);

%% Naive subgradient descent
theta = sparse(zeros(size(kq,1),1)); lambda = 0.55*log(m)/sqrt(np);
tic
step = 1; slength = inf; iter = 0; fold = inf;
while(slength > 1e-5)
    [f, gt] = LLKLIEP(theta,kp,kq);
    g = zeros(size(gt));
    
    id = abs(theta)>0;
    g(id) = gt(id) + lambda*sign(theta(id));
    id = theta==0 & gt > lambda;
    g(id) = gt(id) - lambda;
    id = theta==0 & gt < -lambda;
    g(id) = gt(id) + lambda;
    theta = theta - step*g./(iter+1);
    slength = step*norm(g)./(iter+1);
    fdiff = abs(f - fold);
    
    %display some stuffs
    if iter > 5000
        disp('max iteration reached.')
        break;
    else
        iter = iter+1;
        fdiff = abs(f - fold);
        fold = f;
        if ~mod(iter,100)
            disp(sprintf('%d, %.5f, %.5f, %.5f, nz: %d',...
                iter, slength,fdiff,full(fold),sum(theta(1:end-m)~=0)))
        end
    end
end
toc

%% visualize
tt = ones(m); tt = triu(tt,1); tt=tt(:); idx = tt~=0;
Delta = zeros(1,m*m);
Delta(idx) = theta(1:end-m); Delta = reshape(Delta,m,m);
Delta = Delta + Delta';

hh = figure; spy(thetaP - thetaQ, 64, 'r'); hold on;
spy(abs(Delta), 18); title(sprintf('lambda = %.5f',lambda))
h = legend('ground truth', 'detected'); h.Location='southeast';

%% Function LLKLIEP
%%
%
%   function [l,g,h] = LLKLIEP(theta,kP,kQ)
%   l = -mean(theta'*kP,2) + log(sum(exp(theta'*kQ),2));
%
%   N_q = sum(exp(theta'*kQ),2);
%   g_q = exp(theta'*kQ)./ N_q;
%   g = -mean(kP,2) + kQ*g_q';
%
%   % hessian
%   if nargout>2
%       HH = diag(g_q) - g_q'*g_q;
%       h = kQ*HH*kQ';
%   end
%   end
%

%% Function kernel_linear
%%
%
%   function [k] = kernel_linear(x)
%   [d n] = size(x); k = [];
%   tt = ones(d); tt = triu(tt,1); tt=tt(:); idx = tt~=0;
%   for i = 1:n
%       t = x(:,i) * x(:,i)';
%       t = triu(t,1); t = t(:);
%       k = [k, t(idx)];
%   end
%   k = [k;x.^2];
%   end
%
