function [alpha,score]=KLIEP_learning(mean_X_de,X_nu)
  
  [n_nu,nc]=size(X_nu);
  
  max_iteration=100;
  epsilon_list=10.^[3:-1:-3];
  c=sum(mean_X_de.^2);
  alpha=ones(nc,1);
  [alpha,X_nu_alpha,score]=KLIEP_projection(alpha,X_nu,mean_X_de,c);

  for epsilon=epsilon_list
    for iteration=1:max_iteration
      alpha_tmp=alpha+epsilon*X_nu'*(1./X_nu_alpha);
      [alpha_new,X_nu_alpha_new,score_new]=...
	  KLIEP_projection(alpha_tmp,X_nu,mean_X_de,c);
      if (score_new-score)<=0
        break
      end
      score=score_new;
      alpha=alpha_new;
      X_nu_alpha=X_nu_alpha_new;
    end
  end
