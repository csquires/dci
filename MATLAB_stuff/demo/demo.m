%% Generate Data Set
clear; rng default; m = 1; np = 50; nq = 50; nt = 1000;
xp = randn(m,np); xq = randn(m,nq);
kp = kernel_linear(xp); kq = kernel_linear(xq);
figure; subplot(1,2,1); hist(xp,10); subplot(1,2,2); hist(xq,10)

%% Naive gradient descent
% KLIEP by naive gradient descent
theta = zeros(size(kq,1),1);
tic
step = .01; slength = inf; iter = 0; fold = inf;
while(slength > 1e-5)
    % computing gradient
    [f, g] = LLKLIEP(theta,kp,kq);
    
    theta = theta - step*g./(iter+1);
    slength = step*norm(g)./(iter+1);
    
    %display some stuffs
    if iter > 1000
        disp('max iteration reached.')
        break;
    else
        iter = iter+1;
        fdiff = abs(f - fold);
        fold = f;
        if ~mod(iter,100)
            disp(sprintf('%d, %.5f, %.5f, %.5f, nz: %d',...
                iter, slength,fdiff,full(fold),sum(theta~=0)))
        end
    end
end
toc

%% Error metric
%%
% $$\int q(x) (g(x;\theta) -r(x))^2 ~\mathrm{d}x \approx \frac{1}{n_q} \sum_{i=1}^{n_q} g(x^{(i)};\theta)^2 - \frac{2}{n_p} \sum_{i=1}^{n_p} g(x^{(i)};\theta) + C$$

xtq = randn(m,nt); ktq = kernel_linear(xtq);
gq = exp(theta'*ktq - log(mean(exp(theta'*kq),2)));

xtp = randn(m,nt); ktp = kernel_linear(xtp);
gp = exp(theta'*ktp - log(mean(exp(theta'*kq),2)));
err = mean(gq.^2) - 2*mean(gp);
disp(sprintf('error - C = %.5f', err))

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
