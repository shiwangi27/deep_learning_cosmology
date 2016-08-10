function whiten_images(img_dir, results_dir, N, mean_var_file)
%Usage: for tiny images N = 32;
% whitening images using bruno's sparsenet code

if ~isnumeric(N)
	N = str2num(N);
end

% loads a variable called variance to be passed to the recursive call
load(mean_var_file);

[fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);

whiten_images_recursive_call(img_dir, results_dir, N, filt, variance);
end
