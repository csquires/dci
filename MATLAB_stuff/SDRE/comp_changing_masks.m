function [mask_changing] = comp_changing_masks(adj_nu, adj_de, mc_2)
%find changing edges
mask = logical(adj_nu - adj_de);
mask_changing = zeros(1,size(mc_2,2));
[ii,jj] = find(triu(mask) == 1);
for i = 1:size(mc_2,2)
    t = mc_2(:,i);
    for t1 = [ii(1:end)';jj(1:end)']
        if t(t1(1)) == 1 && t(t1(2)) == 1
            mask_changing(i) = 1;
        end
    end
end
mask_changing = logical(mask_changing);
end