function [n, m, M2] = compute_mean_std_recursive_call(img_dir, results_dir, n, m, M2)
['in recursive call']
contents = dir(img_dir);
for i = 1:length(contents)
    item_name = contents(i).name;
    if(~isempty(strmatch('.',item_name)) || ~isempty(strmatch('..',item_name)))
		continue;
	end
	if isdir([img_dir item_name])
		[img_dir item_name]
		[n, m, M2] = compute_mean_std_recursive_call(...
						[img_dir item_name '/'],...
						[results_dir item_name '/'],...
						n, m, M2);
	else
		% loads a vector of grayscale values - I_tiny
		try
			['i = ', i]
			['going to load ' img_dir item_name]
    		load([img_dir item_name]);
			['loaded ' img_dir item_name]
			if (isempty(I_tiny))
				['was empty']
				continue;
			end
		catch ME
			[img_dir item_name ' doesnt exist.']
			continue;
		end
		% update the mean and std
        n = n + 1;
        delta = I_tiny - m;
        m = m + delta/n;
        M2 = M2 + delta'*(I_tiny - m);
	end
end
end
