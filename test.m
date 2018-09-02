function test()
    
run ~/src/vlfeat/toolbox/vl_setup
run ~/src/matconvnet/matlab/vl_setupnn ;
addpath('/Users/vedaldi/Documents/Publications/2014/mahendran15understanding/code');

im = imread('data/signs-sample-image.jpg');
im = im2single(im);
%im = mean(im,3);

net = hog_net(8);
net = vl_simplenn_tidy(net);

res = vl_simplenn(net,im);

vl_imarraysc(res(1).x) ;


for i=[8]
    figure(i);clf;
    file = sprintf('/tmp/x%d.txt',i) ;
    [a,x,x_] = cmp(res(i).x, file) ;
    imagesc(vl_imarraysc(a)) ;
end

figure(100);clf;
plot([x(1:200)' x_(1:200)'])
disp(mean(abs(x(:)-x_(:))))


function [a,x,x_] = cmp(x,file)
sz = [size(x),1] ;
x_ = load(file) ;
h = sz(1) ;
w = sz(2) ;
c = sz(3) ;
n = sz(4) ;
x_ = reshape(x_, [w,h,c,n]) ;
x_ = permute(x_, [2 1 3 4]) ;
a = cat(2, x, x_, x-x_) ;

