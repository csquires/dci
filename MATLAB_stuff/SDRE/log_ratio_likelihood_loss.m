function [l,g] = log_ratio_likelihood_loss(theta,lambda,k_nu,k_de)
n_theta = norm(theta(:),2);
l = -mean(theta'*k_nu - log(mean(exp(theta'*k_de)))) + lambda*n_theta^2;

p_model = exp(theta'*k_de - log(mean(exp(theta'*k_de))));
g = - (mean(k_nu,2) - ...
    mean(k_de .* repmat(p_model,size(theta,1),1),2)) + 2*lambda*theta(:);
end