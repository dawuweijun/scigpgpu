// ====================================================================
// This file is released under the 3-clause BSD license. See COPYING-BSD.
// ====================================================================
sci_src_dir = get_absolute_file_path("builder_src.sce");
// ====================================================================
languages = ["c", "cpp"];
if WITH_CUDA then
  languages = ["cu", languages];
end
tbx_builder_src_lang(languages, sci_src_dir);
// ====================================================================
clear tbx_builder_src_lang;
clear sci_src_dir;
// ====================================================================
