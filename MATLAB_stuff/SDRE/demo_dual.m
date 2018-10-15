clear
%dual implementation
seed = 1;
rng(seed)

%generate samples
n_pq = 500;
n_test = 3000;
d = 50;
[x1, adj_nu] = random_sparse_sparse2(seed,d,n_pq+n_test,'nu',false);
[x2, adj_de] = random_sparse_sparse2(seed,d,n_pq+n_test,'de',false);
idx = randperm(n_pq+n_test);
x_nu = x1(idx(1:n_pq),:)';
x_de = x2(idx(1:n_pq),:)';
x_nu_t = x1(idx(n_pq+1:end),:)';
x_de_t = x2(idx(n_pq+1:end),:)';

%method parameters
lambda1 = .2; lambda2_list = logspace(1,-2,20);
b = 6;
n = size(x_nu,2);
d = size(x_nu,1);
n_test_nu = size(x_nu_t,2);

%masks for univariate and pairwise cliques
mc = logical(cliques12(d));
c = size(mc,2);
mask = [];
k = kernelize_poly(x_nu, x_de, [x_nu_t,x_de_t], mc, b);
for i = 1:floor(size(k,1)/b)
    mask = [mask,ones(1,b)*i];
end
mc_2 = mc(:,sum(mc,1)==2);
%find changing edges
mask_changing = comp_changing_masks(adj_nu, adj_de, mc_2);

%% run method
regpath_dual = zeros(mask(end),length(lambda2_list));
regpath_primal = zeros(mask(end),length(lambda2_list));

h= mean(k(:,1:n),2);
H= k(:,n+1:2*n);

alpha_star = ones(n,1)/n;
lambda2_list = lambda2_list(end:-1:1);
LL_t = [];

tic
%% compute the regularization path
for i = 1:length(lambda2_list)
    lambda2 = lambda2_list(i);
    
    [~,f_star,alpha_star] = estimate_log_density_ratio_dual...
        (alpha_star,H,h,mask,lambda1,lambda2);
    
    theta_prime = zeros(b,c);
    for j = 1:mask(end)
        x = h(mask==j) - H(mask==j,:)*alpha_star;
        theta = norm(x);
        if theta <= lambda2
            theta = zeros(size(x));
        else
            theta = (1/lambda1 - lambda2/lambda1/norm(x))*x;
        end
        regpath_dual(j,i) = norm(theta);
        theta_prime(:,j) = theta;
    end
    
    %compute the HOLL
    lr_hat = comp_log_ratio(theta_prime(:),k(:,n+1:n+n_test_nu),k(:,n+n_test_nu+1:end));
    LL_t = [LL_t, mean(lr_hat)];
end
elapsed_time = toc;
display(sprintf('total elapsed time: %.2f', elapsed_time))
%% plot and save
h_LL = figure('Visible','on');
semilogx(lambda2_list, LL_t,'linewidth',4); grid on; hold on;
[~,idx_maxLL]=max(LL_t);
text(lambda2_list(idx_maxLL),LL_t(idx_maxLL),'Maximum','HorizontalAlignment','center',... 
	'BackgroundColor',[.7 .9 .7])
h=scatter(lambda2_list(idx_maxLL),LL_t(idx_maxLL));
hChildren = get(h, 'Children');
set(hChildren, 'Markersize', 32)
title('Hold-out Likelihood')

reg_pair_path = regpath_dual(d+1:end,:)';
t_old = zeros(size(reg_pair_path,1),1);
for i = 1:length(lambda2_list)
    t = reg_pair_path(:,i);
    t(t_old - t > 0) = 0;
    reg_pair_path(:,i) = t;
    t_old = t;
end

h=figure('Visible','on');
if sum(mask_changing) ~=0
    h1=loglog(lambda2_list, reg_pair_path(:,mask_changing),'k--','linewidth',2);
    hold on;
end
h2=loglog(lambda2_list, reg_pair_path(:,~mask_changing),'r');
yLimits = get(gca,'YLim');
h3=plot([lambda2_list(idx_maxLL),lambda2_list(idx_maxLL)],...
    [yLimits(1),yLimits(2)],'linewidth',4);
hold on;
yLimits = get(gca,'YLim');
legend([h1(1),h2(1),h3(1)],'Changed', 'Unchanged',  '\lambda_2 picked')
xlabel('\lambda_2')
ylabel('||\theta_{i,j}||')
title(sprintf('maximal HOLL: %f',LL_t(idx_maxLL)))

