n = 7;  // a n x n interpolation grid
x = linspace(0,2*%pi,n); y = x;
z = cos(x')*cos(y);
C = splin2d(x, y, z, "periodic");

// now evaluate on a bigger domain than [0,2pi]x [0,2pi]
m = 80; // discretisation parameter of the evaluation grid
xx = linspace(-0.5*%pi,2.5*%pi,m); yy = xx;
[XX,YY] = ndgrid(xx,yy);

zz1 = interp2d(XX,YY, x, y, C, "C0");
zz2 = interp2d(XX,YY, x, y, C, "by_zero");

gpuzz1 = gpuInterp2d(XX,YY, x, y, C, "C0");
dzz1 = gpuGetData(gpuzz1); gpuFree(gpuzz1);
gpuzz2 = gpuInterp2d(XX,YY, x, y, C, "by_zero");
dzz2 = gpuGetData(gpuzz2); gpuFree(gpuzz2);

//************ show all result ************/
if or(getscilabmode() == ["NW";"STD"]) then

    scf();
    clf()
    subplot(2,2,1)
    plot3d(xx, yy, zz1, flag=[2 6 4])
    xtitle("extrapolation using CPU with the C0 outmode")
    subplot(2,2,3)
    plot3d(xx, yy, zz2, flag=[2 6 4])
    xtitle("extrapolation using CPU with the by_zero outmode")

    subplot(2,2,2)
    plot3d(xx, yy, dzz1, flag=[2 6 4])
    xtitle("extrapolation using GPU with the C0 outmode")
    subplot(2,2,4)
    plot3d(xx, yy, dzz2, flag=[2 6 4])
    xtitle("extrapolation using GPU with the by_zero outmode")

    show_window()
end
