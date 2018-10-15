%% Load Data Set
n_samples = 1000;
p = 10;
n_folds = 10;
fold_samples = 150;
folder = ['../../data/non-gaussian-low-dim-n=', num2str(n_samples), '/'];
lambdas = [.1, .5, 1, 2, 4, 8];
for n_pair=0:99
    disp('========');
    disp(['pair', num2str(n_pair)]);
    disp('========');
    m = 10;
    xp = dlmread([folder, 'Xs/X1_', num2str(n_pair), '.txt'], '\t')';
    xq = dlmread([folder, 'Xs/X2_', num2str(n_pair), '.txt'], '\t')';
    for n_fold=0:n_folds
        xp_fold = xp(:, randsample(n_samples, fold_samples));
        xq_fold = xq(:, randsample(n_samples, fold_samples));
        kp = kernel_linear(xp_fold); kq = kernel_linear(xq_fold);
        np = size(xp_fold, 2);

        for lambda_=lambdas
            disp(lambda_);
            theta = sparse(zeros(size(kq,1),1));
            lambda = lambda_*log(m)/sqrt(np);

            step = 1; slength = inf; iter = 0; fold = inf;
            while(slength > 1e-5)
                [f, gt] = LLKLIEP(theta,kp,kq);

                % soft thresholding
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
                    disp('max iteration reached.');
                    break;
                else
                    iter = iter+1;
                    fdiff = abs(f - fold);
                    fold = f;
    %                 if ~mod(iter,100)
    %                     disp(sprintf('%d, %.5f, %.5f, %.5f, nz: %d',...
    %                         iter, slength,fdiff,full(fold),sum(theta(1:end-m)~=0)));
    %                 end
                end
            end
            foldername = [folder, 'kliep_subsampling/lambda=', sprintf('%.3f', lambda_), '/'];
            mkdir(foldername);
            dlmwrite([foldername, 'K_', num2str(n_pair), '_', num2str(n_fold), '.txt'], theta, '\t');
        end % end lambdas
    end % end folds
end % end pairs