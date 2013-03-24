stacksize('max');
gpuDeviceMemInfo()

a = -50; b = 50;
x = linspace(a,b,100)';
y = sinc(x);
dk = splin(x,y);  // not_a_knot
df = splin(x,y, "fast");
xx = linspace(a,b,400000)';
tic();[yyk, yy1k, yy2k] = interp(xx, x, y, dk);toc()
tic();[yyf, yy1f, yy2f] = interp(xx, x, y, df);toc()

tic();[d_yyk, d_yy1k, d_yy2k] = gpuInterp(xx, x, y, dk);dyyk = gpuGetData(d_yyk);dyy1k = gpuGetData(d_yy1k);dyy2k = gpuGetData(d_yy2k);toc()
tic();[d_yyf, d_yy1f, d_yy2f] = gpuInterp(xx, x, y, df);dyyf = gpuGetData(d_yyf);dyy1f = gpuGetData(d_yy1f);dyy2f = gpuGetData(d_yy2f);toc()

gpuFree(d_yyk);
gpuFree(d_yy1k);
gpuFree(d_yy2k);
gpuFree(d_yyf);
gpuFree(d_yy1f);
gpuFree(d_yy2f);

gpuDeviceMemInfo()

// show result
if or(getscilabmode() == ["NW";"STD"]) then

    scf();

    // display host result
    clf()
    subplot(3,2,1)
    plot2d(xx, [yyk yyf])
    plot2d(x, y, style=-9)
    legends(["not_a_knot spline","fast sub-spline","interpolation points"], [1 2 -9], "ur",%f)
    xtitle("spline interpolation CPU")
    subplot(3,2,3)
    plot2d(xx, [yy1k yy1f])
    legends(["not_a_knot spline","fast sub-spline"], [1 2], "ur",%f)
    xtitle("spline interpolation (derivatives) CPU")
    subplot(3,2,5)
    plot2d(xx, [yy2k yy2f])
    legends(["not_a_knot spline","fast sub-spline"], [1 2], "lr",%f)
    xtitle("spline interpolation (second derivatives) CPU")

    // display device result
    subplot(3,2,2)
    plot2d(xx, [dyyk dyyf])
    plot2d(x, y, style=-9)
    legends(["not_a_knot spline","fast sub-spline","interpolation points"], [1 2 -9], "ur",%f)
    xtitle("spline interpolation GPU")
    subplot(3,2,4)
    plot2d(xx, [dyy1k dyy1f])
    legends(["not_a_knot spline","fast sub-spline"], [1 2], "ur",%f)
    xtitle("spline interpolation (derivatives) GPU")
    subplot(3,2,6)
    plot2d(xx, [dyy2k dyy2f])
    legends(["not_a_knot spline","fast sub-spline"], [1 2], "lr",%f)
    xtitle("spline interpolation (second derivatives) GPU")
end
