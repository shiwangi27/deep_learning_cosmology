function [] = compute_mean_std(img_dir, results_dir, N)
%Usage: for tiny images N = 32;
% Online mean and variance computation according to Knuth:
% http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

if ~isnumeric(N)
	N = str2num(N);
end

n = 0;
m = zeros(N*N,1);
M2 = 0;

['going to recursive call']
[n, m, M2] = compute_mean_std_recursive_call(img_dir, results_dir, n, m, M2)
['end recursive call']
variance = M2/(n - 1);
% save the mean and std
save([results_dir 'data_mean_variance.mat'], 'm', 'variance');
end
