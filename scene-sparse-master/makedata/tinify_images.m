function [] = tinify_images(img_dir, results_dir, N, image_format)
%Usage: for tiny images N = 32;

N = str2num(N);

% list all letter directories
letter_directories = dir(img_dir);
% for each letter directory
for i = 1:length(letter_directories)
    letter_dir = letter_directories(i).name
    if(~isempty(strmatch('.',letter_dir)))
		continue;
	end
    if (~exist([results_dir letter_dir],'dir'))
    	mkdir([results_dir letter_dir]);
    end
	image_directories = dir([img_dir letter_dir '/']);
	for k = 1:length(image_directories)
		current_dir = image_directories(k).name 
    	if(~isempty(strmatch('.',current_dir)))
			continue;
		end
		current_write_path = [results_dir letter_dir '/' current_dir];
        if (~exist(current_write_path,'dir'))
            mkdir(current_write_path);
       	end
        current_path = [img_dir letter_dir '/' current_dir '/'];
        % list all images in this directory
       	images = dir([current_path '/*' image_format]);
        for j = 1:length(images)
            current_image = images(j).name;
            I = im2double(imread([current_path current_image]));
			[~,~,color_channels] = size(I);
			if color_channels > 1
				I = rgb2gray(I);
			end
            I_tiny = imresizecrop(I, [N N]);
            %imwrite(I_tiny, [current_write_path '/' current_image]);
			I_tiny = I_tiny(:);
			% lets save it not as an image but as a vector or the grayscale values 
			save([current_write_path '/' current_image '.mat'], 'I_tiny');
		end
    end
end
end
