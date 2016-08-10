function whiten_images_recursive_call(img_dir, results_dir, N, filt, variance)
contents = dir(img_dir);
for i = 1:length(contents)
    item_name = contents(i).name;
    if(~isempty(strmatch('.',item_name)) || ~isempty(strmatch('..',item_name)))
		continue;
	end
	if isdir([img_dir item_name])
    	if (~exist([results_dir item_name],'dir'))
    		mkdir([results_dir item_name]);
		end
		whiten_images_recursive_call(...
						[img_dir item_name '/'],...
						[results_dir item_name '/'],...
						N, filt, variance);
	else
		% loads a vector of grayscale values - I_tiny
    	load([img_dir item_name]);
		image = reshape(I_tiny, N, N);
    	If=fft2(image);
    	imagew=real(ifft2(If.*fftshift(filt)));
    	I_tiny=reshape(imagew,N^2,1);
		I_tiny=sqrt(0.1)*I_tiny/sqrt(variance);	
		% save the image back as a vector of (whitened) grayscale values	
		save([results_dir item_name], 'I_tiny');
	end
end
end
