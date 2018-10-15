
for i=0:1
    X1_fn = '../data/skeleton_inference/Xs/X1_' + string(i) + '.txt';
    X2_fn = '../data/skeleton_inference/Xs/X2_' + string(i) + '.txt';
    X1 = importdata(X1_fn, '\t')';
    X2 = importdata(X2_fn, '\t')';
    [a, b] = KLIEP(X1, X2, [], .1);
end
