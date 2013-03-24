stacksize('max');
a=rand(4000,4000);
b=rand(4000,4000);
exec scalarProdFunc.sce
tic(); res=scalarProduct(a,b); toc()
res
tic(); res=trace(a'*b); toc()
res

