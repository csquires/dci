clear
% main portal for ratio structure learning
seed= 1;
rng(seed)

%generate samples
n_pq = 500;
d = 50;
n_test = 3000;
[x1, adj_nu] = random_sparse_sparse2(seed,d,n_pq+n_test,'nu',false);
[x2, adj_de] = random_sparse_sparse2(seed,d,n_pq+n_test,'de',false);
idx = randperm(n_pq+n_test);
x_nu = x1(idx(1:n_pq),:)';
x_de = x2(idx(1:n_pq),:)';
x_nu_t = x1(idx(n_pq+1:end),:)';
x_de_t = x2(idx(n_pq+1:end),:)';

%sample parameters
d = size(x_nu,1);
n = size(x_nu,2)+size(x_de,2);
n_nu = size(x_nu,2);
n_test_nu = size(x_nu_t,2);

%method parameters
lambda_list = logspace(1,-2,20);
b = 6;
lambda_1 = 0.1;

flag = false;
clk = clock;

%masks for univariate and pairwise cliques
mc = cliques12(d);
mc_1 = mc(:,sum(mc,1)==1);
mc_2 = mc(:,sum(mc,1)==2);
mc = [mc_1, mc_2];
mc = logical(mc);
c = size(mc,2);

%find changing edges
mask_changing = comp_changing_masks(adj_nu, adj_de, mc_2);
figure
spy(adj_nu- adj_de)

%% run method
theta_init = zeros(b,c);
k = kernelize_poly(x_nu,x_de,[x_nu_t,x_de_t],mc,b);
reg_path = [];
LL = [];
LL_t = [];
theta_path = zeros(b,c,length(lambda_list));

tic
%% compute the regularization path
for lambda = lambda_list
    display(sprintf('lambda2:%f',lambda))
    theta_init = estimate_log_density_ratio(theta_init,k(:,1:n),...
        n_nu,b,d,c,lambda_1,lambda);
    %compute the HOLL
    lr_hat = comp_log_ratio(theta_init(:),k(:,n+1:n+n_test_nu),k(:,n+n_test_nu+1:end));
    LL_t = [LL_t, mean(lr_hat)];
	display(mean(lr_hat))
    
    %store the parameter & reg path
    theta_path(:,:,lambda_list==lambda) = theta_init;
    reg_path = [reg_path;sqrt(sum(theta_init.^2,1))];
    
    %testing if we get the correct pattern
    norm2_theta_pairwise = reg_path(end,d+1:end);
    t1 = sum(norm2_theta_pairwise(~mask_changing)>1e-6) == 0;
    t2 = sum(norm2_theta_pairwise(mask_changing)<1e-6) == 0;
    
    if t1&&t2
        display('SUCCESS!')
        flag = true;
%         return;
    end
end
elapsed_time = toc;
display(sprintf('total elapsed time: %.2f', elapsed_time))
%% plot and save
h_LL = figure('Visible','on');
semilogx(lambda_list, LL_t,'linewidth',4); grid on; hold on;
[~,idx_maxLL]=max(LL_t);
text(lambda_list(idx_maxLL),LL_t(idx_maxLL),'Maximum','HorizontalAlignment','center',... 
	'BackgroundColor',[.7 .9 .7])
h=scatter(lambda_list(idx_maxLL),LL_t(idx_maxLL));
hChildren = get(h, 'Children');
set(hChildren, 'Markersize', 32)
title('Hold-out Likelihood')

reg_pair_path = reg_path(:,d+1:end);
t_old = zeros(size(reg_pair_path,1),1);
for i = 1:length(lambda_list)
    t = reg_pair_path(:,i);
    t(t_old - t > 0) = 0;
    reg_pair_path(:,i) = t;
    t_old = t;
end

h=figure('Visible','on');
if sum(mask_changing) ~=0
    h1=loglog(lambda_list, reg_pair_path(:,mask_changing),'k--','linewidth',2);
    hold on;
end
h2=loglog(lambda_list, reg_pair_path(:,~mask_changing),'r');
hold on;
yLimits = get(gca,'YLim');
h3=plot([lambda_list(idx_maxLL),lambda_list(idx_maxLL)],...
    [yLimits(1),yLimits(2)],'linewidth',4);
legend([h1(1),h2(1),h3(1)],'Changed', 'Unchanged', '\lambda_2 picked')
xlabel('\lambda_2')
ylabel('||\theta_{i,j}||')
title(sprintf('maximal HOLL: %f',LL_t(idx_maxLL)))

saveas(h,'regularization_path.png');
saveas(h_LL,'holdout_likelihood.png');
save('solution_path','reg_path','LL_t','theta_path');
