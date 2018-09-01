import os
from PIL import Image

prefix = os.path.join('data', 'tmp', 'TrainIJCNN2013') ;

[names,x1,y1,x2,y2,labels] = textread(fullfile(prefix, 'gt.txt'), ...
  '%s%d%d%d%d%d', 'headerlines', 1, 'delimiter', ';') ;
boxes = [x1, y1, x2, y2]'+1 ;

images = fullfile(prefix, names) ;
patches = {} ;
for j = 1:numel(images)
  t = imread(images{j}) ;
  t = im2single(t) ;
  t = imcrop(t, [x1(j) y1(j) x2(j)-x1(j)+1 y2(j)-y1(j)+1]) ;
  t = imresize(t, [64 64]) ;
  patches{j} = t ;
  [~,base,~] = fileparts(images{j}) ;
  images{j} = fullfile('data', 'signs', [base '.jpeg']) ;
end
patches = cat(4, patches{:}) ;