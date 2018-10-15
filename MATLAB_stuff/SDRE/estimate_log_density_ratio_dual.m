function [theta_star, f_star, alpha_star] = estimate_log_density_ratio_dual...
    (alpha_init,H,h,mask,lambda1,lambda2)
options.maxIter = 1000;
options.optTol = 1e-9;
options.TolFun = 1e-9;
options.verbose = 0;

fObj = @(alpha)f(alpha,H,h,mask,lambda1, lambda2);
[alpha_star, f_star] = minConf_SPG(fObj,alpha_init, ...
    @(alpha)projectSimplex(alpha),options);

theta_star = [];
% for j = 1:mask(end)
%     x = h(mask==j) - H(mask==j,:)*alpha_star;
%     t = norm(x);
%     if t <= lambda2
%         theta_star = [theta_star;zeros(size(x))];
%     else
%         theta_star = [theta_star; (1/lambda1 - lambda2/lambda1/norm(x))*x];
%     end
% end

end