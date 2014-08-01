prefix = 'data/tmp/TrainIJCNN2013' ;
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
end
patches = cat(4, patches{:}) ;

train = unique(names) ;
train = train(randperm(numel(train))) ;
train = train(1:400) ;
train = ismember(names, train) ;
test = ~train ;

trainImages = unique(images(train)) ;
trainBoxes = boxes(:, train) ;
trainBoxImages = images(train) ;
trainBoxLabels = labels(train) ;
trainBoxPatches = patches(:,:,:,train) ;

testImages = unique(images(test)) ;
testBoxes = boxes(:, test) ;
testBoxImages = images(test) ;
testBoxLabels = labels(test) ;
testBoxPatches = patches(:,:,:,test) ;

save('data/signs-data.mat', ...
  'trainImages', ...
  'trainBoxes', ...
  'trainBoxImages', ...
  'trainBoxLabels', ...
  'trainBoxPatches', ...
  'testImages', ...
  'testBoxes', ...
  'testBoxImages', ...
  'testBoxLabels', ...
  'testBoxPatches') ;
