function [k] = kernelize_poly(x_nu,x_de,x_test,mc,b)
% kernelization of clique potentials.
% x_nu, x_de samples from numerator and denominator densities
% mc the mask of clique potentials
% b the number of basis
x = [x_nu, x_de, x_test];
[d,n] = size(x);
c = size(mc,2);

k = zeros(b*c,n);
for i = 1:c
    x_s = x(mc(:,i), :);
    if i<=d
        x_s = [x_s;zeros(1,n)];
    end
    k(b*(i-1)+1:b*i,:) = kernel_poly(x_s,b);
end
end

function [k] = kernel_poly(x,b)
n = size(x,2);
k = [];

degree = (-1 + sqrt((1+8*b)))/2 -1;

for i = 1:degree
    for j = 0:i
        k = [k; (x(1,:).^j).*(x(2,:).^(i-j))];
    end
end
k = [k;ones(1,n)];

end