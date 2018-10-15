function [f_val, g_val] = f(alpha, H, h, mask, lambda1, lambda2)
% The dual objective for LL-KLIEP
n = size(H,2);
b = size(H,1)/mask(end);
f_val = sum(alpha.*log(alpha));
g_val = log(alpha)+1;
for i = 1:mask(end)
    %fast! no indexing...
    H_t = H((i-1)*b+1:i*b, :);
    h_t = h((i-1)*b+1:i*b, :);
    % slow! indexing...
    %         H_t = H(mask == i,:);
    %         h_t = h(mask == i);
    [f, g] = f_elastic_star(alpha, H_t, h_t, lambda1, lambda2);
    f_val = f_val + f;
    g_val = g_val + g;
end
end

function [f_val, g_val] = f_elastic_star(alpha, H_t, h_t, lambda1, lambda2)
x_t = h_t - H_t*alpha;
if norm(x_t) <= lambda2
    f_val = 0;
else
    f_val = ((x_t'*x_t - lambda2^2)/(2*lambda1)) - ...
        ((lambda2*norm(x_t) - lambda2^2)/lambda1);
end

if norm(x_t) <= lambda2
    g_val = zeros(size(alpha));
else
    g_val = (lambda2*H_t'*(x_t/norm(x_t)))/lambda1 - ...
        (H_t'*x_t)/lambda1;
end
end