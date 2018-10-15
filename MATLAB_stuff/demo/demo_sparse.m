%% Load Data Set
function nothing = run_kliep(folder, n_samples, p, lambdas, npairs)
    for n_pair=0:(npairs-1)
        disp('========');
        disp(['pair', num2str(n_pair)]);
        disp('========');
        m = 10;
        xp = dlmread([folder, 'pair', num2str(n_pair), '/parameters/B1.txt'], '\t')';
        xq = dlmread([folder, 'pair', num2str(n_pair), '/parameters/B2.txt'], '\t')';
        kp = kernel_linear(xp); kq = kernel_linear(xq);
        np = size(xp, 2);

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
            foldername = [folder, 'pair', num2str(n_pair), '/samples_n=', num2str(n_samples), '/results/kliep/lambda=', sprintf('%.3f', lambda_), '/'];
            mkdir(foldername);
            dlmwrite([foldername, 'K', num2str(n_pair), '.txt'], theta, '\t');
        end
    end
end
