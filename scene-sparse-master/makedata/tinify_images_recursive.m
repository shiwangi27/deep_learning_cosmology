function [] = tinify_images_recursive(img_dir, results_dir, N)
%Usage: for tiny images N = 32;

if ~isnumeric(N)
	N = str2num(N);
end

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
		tinify_images_recursive([img_dir item_name '/'], [results_dir item_name '/'], N);
	else
    	I = im2double(imread([img_dir item_name]));
		[~,~,color_channels] = size(I);
		if color_channels > 1
			I = rgb2gray(I);
		end
        I_tiny = imresizecrop(I, [N N]);
		I_tiny = I_tiny(:);
		% lets save it not as an image but as a vector or the grayscale values 
		save([results_dir item_name '.mat'], 'I_tiny');
	end
end
end
