function [mc] = cliques12(d)
    mc = eye(d);
    for i = 1:d
        for j = 1:i-1
            t = zeros(d,1);
            t(i) = 1;
            t(j) = 1;
            
            mc = [mc,t];
        end
    end
end