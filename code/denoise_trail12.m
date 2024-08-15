clear all; 
close all;
clc;
load denoise18
x=imread('C:\Users\siddu\Downloads\Project\ds\07.png');
% x=rgb2gray(x);
% x=imresize(x,[256 256]);
 x_noisy=imnoise(x,'salt & pepper',0.1);
% [m n]=size(x_noisy);
% f=zeros(m,n);
% for i=1:m
%   for  j=1:n
%       if x_noisy(i,j)==0||x_noisy(i,j)==255
%     f(i,j)=0;
%       else 
%           f(i,j)=x_noisy(i,j);
%       end
%   end
% end
% figure
% imshow(uint8(f))
net = denoisingNetwork('dncnn');
y = denoiseImage1(x_noisy, denoise18);
z=denoiseImage(x_noisy,net);
figure
subplot(1,2,1)
imshow(x_noisy)
title 'noisy image'
subplot(1,2,2)
imshow(medfilt2(y))
title 'denoised using residual net'
% z=histeq(y);
p=psnr(x,y)
s=ssim(x,medfilt2(y))