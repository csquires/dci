function [alpha,Xte_alpha,score]=KLIEP_projection(alpha,Xte,b,c)
%  alpha=alpha+b*(1-sum(b.*alpha))/c;
  alpha=alpha+b*(1-sum(b.*alpha))*pinv(c,10^(-20));
  alpha=max(0,alpha);
%  alpha=alpha/sum(b.*alpha);
  alpha=alpha*pinv(sum(b.*alpha),10^(-20));
  Xte_alpha=Xte*alpha;
  score=mean(log(Xte_alpha));
