
k=18; 
se=strel('disk',k);
fo=imopen(aa,se);%������
foc1=imclose(fo,se);%������
%figure; imshow(foc,[])
se=strel('disk',k);
fo=imopen(bb,se);%������
foc2=imclose(fo,se);%������
%figure; imshow(foc2,[])
se=strel('disk',k);
fo=imopen(cc,se);%������
foc3=imclose(fo,se);%������
%figure; imshow(foc3,[])
e1=[foc1(:);foc2(:);foc3(:)];
FW=computeFWIoU(e2,e1)



%%  ��ɫ������
if (pre_lab==0)~=[] %exist black class
 %input the original data  use two channel is enough
a1=dataHH(:,:,28);
a4=dataVV(:,:,28);

b=find(a1==0&a4==0); 
c=ones(512,512);
c(b)=0; % mask

f=double(x28_feature); %input predict label
%figure; imshow(f,[])
z=f;
hw=40;

z(b)=0;

d=find(f==0 & c==1);

f0=padarray(f,[hw hw],'symmetric');
for i=1:size(d,1)
    [a,b]=ind2sub([512,512],d(i));
    p=f0(a:a+2*hw-1,b:b+2*hw-1);
    q=p(p~=0);
    z(d(i))=mode(q);
end
figure; imshow(z,[])
end