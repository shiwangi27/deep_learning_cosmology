function [C,X] = get_data_correlation(img_dir, num_images_, img_size_, image_format, results_file)
% usage1: C_man_made = get_data_correlation('/clusterfs/cortex/scratch/shiry/image-net-tiny/man_made/', 12575, [32 32], '.JPEG', '/clusterfs/cortex/scratch/shiry/results/data_correlation/man_made');
% usage2: C_natural = get_data_correlation('/clusterfs/cortex/scratch/shiry/image-net-tiny/natural/', 11214, [32 32], '.JPEG', '/clusterfs/cortex/scratch/shiry/results/data_correlation/natural');

%Converting Strings back to integers
num_images=str2num(num_images_);
img_size=[str2num(img_size_),str2num(img_size_)];

%DEBUG. Printing inputs for sanity check
sprintf('%s',img_dir)
sprintf('%s',image_format)
% create the data matrix where each image will be a column vector
if (length(img_size) == 2)
    M = img_size(1) * img_size(2);
else
    M = img_size(1) * img_size(2) * img_size(3);
end
X = zeros(M,num_images);
idx = 1;

% list all image directories
image_directories = dir(img_dir);
% for each image directory
for i = 1:length(image_directories)
    current_dir = image_directories(i).name
    if(isempty(strmatch('.',current_dir)))
        current_path = [img_dir current_dir '/'];
        % list all images in this directory
        images = dir([current_path '*' image_format])
        for j = 1:length(images)
            current_image = images(j).name
            I = imread([current_path current_image]);
            % convert image to a column vector
            X(:,idx) = reshape(I, [M 1]);
            idx = idx + 1;
        end
    end
end
% compute the row-wise covariance between pixel locations across images in this dataset
C = cov(X');
C_reshape=reshape(diag(C),32,32);
save([results_file '_correlation.mat'],'C','C_reshape');
save([results_file '_images.mat'], 'X');
end
