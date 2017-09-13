clear;clc;
load data;
Fre=48000;
cycle=1/Fre;
length=size(data,2);
time=0:cycle:(length-1)*cycle;
for i=1:size(data,1)
tree=wpdec(data(i,:),3,'db10');
z=1
for j=0:2^3-1
    coef=wprcoef(tree,[3,j]);
    energy(z)=sqrt(sum(abs(coef).^2));
    z=z+1;
end
%wavedata(i,:)=energy./sum(energy)
wavedata(i,:)=mapstd(energy)
end
save('wavedata.mat','wavedata');