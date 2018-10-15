function lr_hat = comp_log_ratio(theta,k_test,k_de)
    lr_hat = theta'*k_test - log(mean(exp(theta'*k_de)));
end