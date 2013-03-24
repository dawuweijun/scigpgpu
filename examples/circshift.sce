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