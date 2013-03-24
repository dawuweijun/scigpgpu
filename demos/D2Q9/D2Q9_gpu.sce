lines(0);

abs_path=get_absolute_file_path("D2Q9_gpu.sce");

function ret=cs(A,m1,m2)
  [m,n]=size(A);
  rettmp=zeros(m,n);
  ret=zeros(m,n);
  for i=1:m
    id=pmodulo(i-m1-1,m)+1;
    rettmp(i,:)=A(id,:);
  end
  for j=1:n
    jd=pmodulo(j-m2-1,n)+1;
    ret(:,j)=rettmp(:,jd);
  end
endfunction

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// % cylinder.m: Channel flow past a cylinderical
// %             obstacle, using a LB method
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// % Lattice Boltzmann sample in Matlab
// % Copyright (C) 2006-2008 Jonas Latt
// % Address: EPFL, 1015 Lausanne, Switzerland
// % E-mail: jonas@lbmethod.org
// % Get the most recent version of this file on LBMethod.org:
// %   http://www.lbmethod.org/_media/numerics:cylinder.m
// %
// % Original implementaion of Zou/He boundary condition by
// % Adriano Sciacovelli (see example "cavity.m")
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// % This program is free software; you can redistribute it and/or
// % modify it under the terms of the GNU General Public License
// % as published by the Free Software Foundation; either version 2
// % of the License, or (at your option) any later version.
// % This program is distributed in the hope that it will be useful,
// % but WITHOUT ANY WARRANTY; without even the implied warranty of
// % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// % GNU General Public License for more details.
// % You should have received a copy of the GNU General Public
// % License along with this program; if not, write to the Free
// % Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
// % Boston, MA  02110-1301, USA.
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// Translated to scilab language by Vincent Lejeune

// start timer
tic();

// GENERAL FLOW CONSTANTS
lx     = 400;      //number of cells in x-direction
ly     = 100;      // number of cells in y-direction
obst_x = lx/5+1;   // position of the cylinder; (exact
obst_y = ly/2+3;   // y-symmetry is avoided)
obst_r = ly/10+1;  // radius of the cylinder
uMax   = 0.1;      // maximum velocity of Poiseuille inflow
Re     = 100;      // Reynolds number
nu     = uMax * 2.*obst_r / Re;  // kinematic viscosity
omega  = 1. / (3*nu+1./2.);      // relaxation parameter
maxT   = 400;  // total number of iterations
tPlot  = 50;      // cycles

// D2Q9 LATTICE CONSTANTS
t  = [4/9, 1/9,1/9,1/9,1/9, 1/36,1/36,1/36,1/36];
cx = [  0,   1,  0, -1,  0,    1,  -1,  -1,   1];
cy = [  0,   0,  1,  0, -1,    1,   1,  -1,  -1];
opp = [ 1,   4,  5,  2,  3,    8,   9,   6,   7];
col = [2:(ly-1)];
in  = 1;   // position of inlet
out = lx;  // position of outlet

[y,x] = meshgrid(1:ly,1:lx); // get coordinate of matrix indices

obst = ...                   // Location of cylinder
    (x-obst_x).^2 + (y-obst_y).^2 <= obst_r.^2;
obst(:,[1,ly]) = 1;    // Location of top/bottom boundary
bbRegion = find(obst); // Boolean mask for bounce-back cells

// INITIAL CONDITION: Poiseuille profile at equilibrium
L = ly-2; y_phys = y-1.5;
ux = 4 * uMax / (L*L) * (y_phys.*L-y_phys.*y_phys);
uy = zeros(lx,ly);


gUX=gpuSetData(ux);
gUY=gpuSetData(uy);
gObst=gpuSetData(obst);
gfIn=list(...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly)...
);

gfEq=list(...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly)...
);

gfOut=list(...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly),...
  gpuAlloc(lx,ly)...
);

gRho = gpuAlloc(lx,ly);
for i=1:9
    cu = 3*(cx(i)*ux+cy(i)*uy);
    tmp = t(i) .* ...
                   ( 1 + cu + 1/2*(cu.*cu) - 3/2*(ux.^2+uy.^2) );
    gfIn(i)=gpuSetData(tmp);
end

bin=gpuBuild(abs_path+"D2Q9_kernels");

fonc=gpuLoadFunction(bin,"compute_rho");
fonc2=gpuLoadFunction(bin,"compute_new_ux_uy");
fonc3=gpuLoadFunction(bin,"boundary_conditions_inlet");
fonc4=gpuLoadFunction(bin,"boundary_conditions_outlet");
fonc5=gpuLoadFunction(bin,"micro_bound_cond_inlet");
fonc6=gpuLoadFunction(bin,"micro_bound_cond_outlet");
fonc7=gpuLoadFunction(bin,"streamingstep1");
fonc8=gpuLoadFunction(bin,"streamingstep2");
fonc9=gpuLoadFunction(bin,"some_aux_calculations");
fonc10=gpuLoadFunction(bin,"bounce");

f=scf();
Matplot(uy');
f.color_map=jetcolormap(256);
e=gce();

// // MAIN LOOP (TIME CYCLES)
for cycle = 1:maxT

//
//     // MACROSCOPIC VARIABLES
     //rho = sum(fIn,'m');
lst=aplat(list(gRho,gfIn,int32(lx*ly) ));

gpuApplyFunction(fonc,lst,1,100,1,ceil(lx*ly/100));

//       tmpx=cx*matrix(fIn,9,lx*ly);
//       tmpy=cy * matrix(fIn,9,lx*ly);
//      ux  = matrix ( tmpx, 1,lx,ly) ./rho;
//      uy  = matrix ( tmpy, 1,lx,ly) ./rho;

 lst=aplat(list(gUX,gUY,gRho,gfIn,int32(lx*ly)));
gpuApplyFunction(fonc2,lst,1,100,1,ceil(lx*ly/100));

     // MACROSCOPIC (DIRICHLET) BOUNDARY CONDITIONS
       // Inlet: Poiseuille profile
     //y_phys = col-1.5;
     //ux(1,in,col) = 4 * uMax / (L*L) * (y_phys.*L-y_phys.*y_phys);
     //uy(1,in,col) = 0;
     //tmp=sum(fIn([1,3,5],in,col),'m') + 2*sum(fIn([4,7,8],in,col),'m');
     //rho(:,in,col) = ones(1,1,98) ./ (1-ux(:,in,col)) .* tmp;


 lst=aplat(list(gUX,gUY,gRho,gfIn,int32(1),int32(99),int32(lx)));
 gpuApplyFunction(fonc3,lst,1,100,1,ceil(ly/100));
       // Outlet: Constant pressure
//     rho(:,out,col) = 1;
//     ux(:,out,col) = -ones(1,1,98) + ones(1,1,98) ./ (rho(:,out,col)) .* ( ...
//         sum(fIn([1,3,5],out,col),'m') + 2*sum(fIn([2,6,9],out,col),'m'));
//     uy(:,out,col)  = 0;


   lst=aplat(list(gUX,gUY,gRho,gfIn,int32(1),int32(99),int32(lx)));
   gpuApplyFunction(fonc4,lst,1,100,1,ceil(ly/100));

//      // MICROSCOPIC BOUNDARY CONDITIONS: INLET (Zou/He BC)
//      fIn(2,in,col) = fIn(4,in,col) + 2/3*rho(:,in,col).*ux(:,in,col);
//      fIn(6,in,col) = fIn(8,in,col) + 1/2*(fIn(5,in,col)-fIn(3,in,col)) ...
//                                      + 1/2*rho(:,in,col).*uy(:,in,col) ...
//                                      + 1/6*rho(:,in,col).*ux(:,in,col);
//      fIn(9,in,col) = fIn(7,in,col) + 1/2*(fIn(3,in,col)-fIn(5,in,col)) ...
//                                      - 1/2*rho(:,in,col).*uy(:,in,col) ...
//                                      + 1/6*rho(:,in,col).*ux(:,in,col);
//

  lst=aplat(list(gUX,gUY,gRho,gfIn,int32(1),int32(99),int32(lx)));
  gpuApplyFunction(fonc5,lst,1,100,1,ceil(ly/100));

     // MICROSCOPIC BOUNDARY CONDITIONS: OUTLET (Zou/He BC)
//      fIn(4,out,col) = fIn(2,out,col) - 2/3*rho(:,out,col).*ux(:,out,col);
//      fIn(8,out,col) = fIn(6,out,col) + 1/2*(fIn(3,out,col)-fIn(5,out,col)) ...
//                                        - 1/2*rho(:,out,col).*uy(:,out,col) ...
//                                        - 1/6*rho(:,out,col).*ux(:,out,col);
//      fIn(7,out,col) = fIn(9,out,col) + 1/2*(fIn(5,out,col)-fIn(3,out,col)) ...
//                                        + 1/2*rho(:,out,col).*uy(:,out,col) ...
//                                        - 1/6*rho(:,out,col).*ux(:,out,col);

  lst=aplat(list(gUX,gUY,gRho,gfIn,int32(1),int32(99),int32(lx)));
  gpuApplyFunction(fonc6,lst,1,100,1,ceil(ly/100));


//      for i=1:9
//         cu = 3*(cx(i)*ux+cy(i)*uy);
//         fEq(i,:,:)  = rho .* t(i) .*( 1 + cu + 1/2*(cu.*cu)  - 3/2*(ux.^2+uy.^2) );
//         fOut(i,:,:) = fIn(i,:,:) - omega .* (fIn(i,:,:)-fEq(i,:,:));
//      end



for i=1:9

  lst=aplat( list(gfIn(i),gfEq(i),gfOut(i),gRho,gUX,gUY,t(i),omega,int32(cx(i)),int32(cy(i)),int32(lx*ly)) );
  gpuApplyFunction(fonc9,lst,1,100,1,ceil(lx*ly/100));
end

//    OBSTACLE (BOUNCE-BACK)
     for i=1:9
      lst=list(gfIn(opp(i)),gfOut(i),gObst,int32(lx*ly));
      gpuApplyFunction(fonc10,lst,1,100,1,ceil(lx*ly/100));
     end

//     STREAMING STEP
//      for i=1:9
// 	tmpmat=matrix(fOut(i,:,:),lx,ly);
// 	tmp=cs(tmpmat,cx(i),cy(i));
//         fIn(i,:,:) = matrix(tmp,1,lx,ly);
//      end
for i=1:9

  lst=aplat(list(gfIn(i),gfOut(i),gRho,int32(cx(i)),int32(cy(i)),int32(lx),int32(ly)));
  gpuApplyFunction(fonc7,lst,1,100,1,ceil(lx/100));

  lst=aplat(list(gfIn(i),gfOut(i),gRho,int32(cx(i)),int32(cy(i)),int32(lx),int32(ly)));
  gpuApplyFunction(fonc8,lst,1,100,1,ceil(lx/100));
end


//    VISUALISATION
		 UX=gpuGetData(gUX);
		 UY=gpuGetData(gUY);
         u = matrix(sqrt(UX.^2+UY.^2),lx,ly);
         u(bbRegion) = %nan;

        img=abs(255*u/max(u));

	if gpuUseCuda() then
		xtitle("Cuda mode : Cycle ="+string(cycle)+"/"+string(maxT)+" Elapsed time : "+string(toc())+"s","X","Y")
	else
		xtitle("OpenCL mode : Time ="+string(cycle)+"/"+string(maxT)+" Elapsed time : "+string(toc())+"s","X","Y")
	end
	if is_handle_valid(e) then
		 e.data=img';
	else
		return
	end
end

