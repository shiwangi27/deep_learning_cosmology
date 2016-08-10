%This script will do the scene sparse algorithm
function [err] = scene_sparse(path)
disp('Starting Execution')
addpath('../fast_sc/code/')

%Input paths
if nargin <1
	path='/clusterfs/cortex/scratch/shiry/scene-sparse/man_made';
end
	%Load data

	load(path)

	%Initiatlize Parameters for SC
	X_orig = X_man_made;
	num_bases = size(X_orig,1)*4; %We start off with a 4x over complete basis
	batch_size = 1000;
	num_iters = 2;
	sparsity_func = 'epsL1'; %The other option is 'L1'
	epsilon = .001;
	beta=.01;
    fname_save = '/clusterfs/cortex/scratch/shiry/scene-sparse/man_made/test.mat';
   

    %Whiten the data 
    
	Binit = [] ; %Inferred coefficients, start with empty
	[B S stat] = sparse_coding(X_orig, num_bases, beta, sparsity_func, epsilon, num_iters, batch_size, fname_save)%, resample_size);
%Save Dictionary
% Shiry will write the save function's path here 
%save('path that shiry will write','B','S','stat');
%Make Image

end
