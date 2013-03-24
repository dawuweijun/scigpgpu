function bOK = isATI()
    bOK = %f;
    if getos() == "Windows" then
        [dyninfo, statinfo] = getdebuginfo();
        version = getversion("scilab");
        if or(version(1:3) >= [5 4 0]) then
            videocard = dyninfo(grep(dyninfo, "Video card #"));
        else
            videocard = dyninfo(grep(dyninfo, "Video card:"));
            videocard = strsubst(videocard, "Video card:", "");
        end
    else
        videocard = unix_g("lspci | grep VGA");
    end

    bOK = grep(convstr(videocard, "u"), "ATI") <> [];
endfunction
