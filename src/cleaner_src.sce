// This file is released under the 3-clause BSD license. See COPYING-BSD.

src_dir = get_absolute_file_path("cleaner_src.sce");

for language = ["c", "cpp", "cu"]
    cleaner_file = src_dir + filesep() + language + filesep() + "cleaner.sce";
    if isfile(cleaner_file) then
        exec(cleaner_file);
        mdelete(cleaner_file);
    end
end

cleaner_file = src_dir + filesep() + 'cu' + filesep() + "cleaner_cu.sce";
if isfile(cleaner_file) then
   exec(cleaner_file);
 end

clear src_dir;
