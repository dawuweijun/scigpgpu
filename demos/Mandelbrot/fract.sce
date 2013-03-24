// Sources : Jonathan Blanchard (http://jblopen.com/node/18)
// Modified by : DIGITEO - 2011 - Cedric DELAMARRE

lines(0);

[a b] = getversion();
if b(2) == "x86" then
    disp('This demo is not available on 32 bits systems.');
else
    abs_path=get_absolute_file_path("fract.sce");

    stacksize('max');
    // init data
    nmax = 800;
    xmin = 0.2675;
    xmax = 0.2685;
    ymin = 0.591;
    ymax = 0.592;
    xsize = 1000;
    ysize = 1000;

    xvect = linspace( xmin, xmax, xsize );
    yvect = linspace( ymin, ymax, ysize );

    // build kernel
    bin=gpuBuild(abs_path+"fract");

    // set clock
    tic();

    // send data on gpu
    d_xvect = gpuSetData(xvect);
    d_yvect = gpuSetData(yvect);
    d_out   = gpuSetData(zeros(xsize,ysize));

    // load function
    Kernel=gpuLoadFunction(bin,"computeFract");

    // perform operation
    lst=list(d_xvect,d_yvect,int32(xsize),int32(ysize),int32(nmax),d_out);
    gpuApplyFunction(Kernel,lst,2,2,500,500);

    // get resultat from gpu
    res = gpuGetData(d_out);

    fact = (gpuMax(d_out)) / 512
    res = res./fact;

    // get time of computation
    toc()

    // show result
    if or(getscilabmode() == ["NW";"STD"]) then
        map = hotcolormap(512);
        cmap = [map(:,3), map(:,2), map(:,1)];

        f = scf();
        f.color_map = cmap;
        f.auto_resize = "off";
        f.axes_size = [1000,1000];
        f.figure_size = [800,800];

        clf();
        a = gca();
        a.margins = [0,0,0,0];
        Matplot(res, "080");
    end
end
