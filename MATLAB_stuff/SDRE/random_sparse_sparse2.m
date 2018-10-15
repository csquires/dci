function [x, adj] = random_sparse_sparse2(seed,d,n,type,npn)

rng(seed);
change = [];
if strcmp(type,'de')
    adj = rand(d,d)<.25;
else
    adj = rand(d,d)<.235;
      
    rng(seed)
    adj_de = rand(d,d)<.25;
    change =  adj_de - adj; 
	rng(seed+1)
end
adj(1:d+1:d*d) = 0;
adj = tril(adj,-1)+tril(adj,-1)';
change = logical(tril(change,-1)+tril(change,-1)');

Theta = zeros(d,d);
Theta(adj==1) = .2;
Theta(change) = -.2;
Theta = Theta + eye(d,d)*2;
x = mvnrnd(zeros(1,d), inv(Theta), n);

if npn
    x = sign(x).*power(abs(x),1/2);
end

end