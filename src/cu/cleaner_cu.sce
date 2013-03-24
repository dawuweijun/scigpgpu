path = pwd();
cu_dir = get_absolute_file_path("cleaner_cu.sce");
cd(cu_dir);
f = ls("*.cu.cpp");
nbr = max(size(f));
if nbr > 0,
	for i = 1:nbr,
		mdelete(f(i,1));
	end
end

cd(path);

clear path;
clear nbr;
clear f;
