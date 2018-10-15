function [theta_hat] = estimate_log_density_ratio(theta_init,k,n_nu,b,d,c,lambda1,lambda2)
% log_density estimates the log density ratio under kullback-leibler loss,
% with group lasso regularization on pairwise potentials, and
% L2-regularization on all parameters
% theta_init initalization of parameters
% k kernel matrix
% n_nu number of samples from numberator density
% b,d,c number of kernel basis, dimensions, and clique potentials
% lambda1,lambda2 regularization paramters for L2-L2 penalty

options.maxIter = 1000;
options.optTol = 1e-7;
options.TolFun = 1e-7;
options.verbose = 1;

% Make Initial Value, adding 'g'
g = zeros(c, 1);
for j = 1:c
    g(j) = norm(theta_init(:,j),2);
end
theta_init = [theta_init(:);g];

funObj_sub = @(theta)log_ratio_likelihood_loss(theta, lambda1, ...
    k(:,1:n_nu),k(:,n_nu+1:end));

% Make Objective and Projection Function
funObj = @(theta)pairwise_loss(theta,b,c,d,lambda2,funObj_sub);
funProj = @(theta)pairwise_projection(theta,b,c,d);

% Solve
theta_hat = minConf_SPG(funObj,theta_init,funProj,options);
theta_hat = reshape(theta_hat(1:b*c),b,c);

end

function [l,g] = pairwise_loss(theta,b,c,d,lambda2,funObj)
[l,g] = funObj(theta(1:b*c));
l = l + sum(lambda2*theta(b*c+1:end));
g = [g; lambda2*ones(c,1)];
end

function [theta] = pairwise_projection(theta,b,c,d)
%project all the cliques not only pairwise
theta_t = reshape(theta(1:b*c),b,c);

g = theta(b*c+1:end);
for i = 1:c
    nt = norm(theta_t(:,i),2);
    avg = (nt + g(i))/2;
    if  nt > abs(g(i))
        
        % renormalize theta(i)
        theta_t(:,i) = theta_t(:,i) * avg / nt;
        g(i) = avg;
    elseif nt <= -g(i)
        theta_t(:,i) = zeros(b,1);
        g(i) = 0;
    end
end

theta = [theta_t(:); g];
end