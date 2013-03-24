// ====================================================================
// Allan CORNET - DIGITEO - 2011
// This file is released into the public domain
// ====================================================================
sci_gateway_dir = get_absolute_file_path('builder_gateway.sce');
// ====================================================================
languages = ['cpp', 'c'];
// ====================================================================
tbx_builder_gateway_lang(languages, sci_gateway_dir);
tbx_build_gateway_loader(languages, sci_gateway_dir);
tbx_build_gateway_clean(languages, sci_gateway_dir);
// ====================================================================
clear languages;
clear tbx_builder_gateway_lang;
clear tbx_build_gateway_loader;
clear tbx_builder_gateway_clean;
clear sci_gateway_dir;
// ====================================================================