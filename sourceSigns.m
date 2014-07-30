
% source test data
prefix = 'data/tmp/GTSRB/Final_Test/Images/' ;
names = dir(fullfile(prefix, '*.ppm')) ;
testImages = fullfile(prefix, {names.name});

[a,b,c,x1,y1,x2,y2] = textread(fullfile(prefix, 'GT-final_test.test.csv'), ...
  '%s%d%d%d%d%d%d', 'headerlines', 1, 'delimiter', ';') ;
  
testBoxes = [x1, y1, x2, y2]'+1 ;

for i =1:numel(testImages) ;
  figure(1) ; clf ;
  imagesc(imread(testImages{i})) ; hold on ; axis equal ;
  vl_plotbox(testBoxes(:, i)) ;
  pause ;
end
keyboard

% source training data
prefix = 'data/tmp/GTSRB/Final_Training/Images/' ;
im = {} ;
lab = [] ;
for i = 0:1
  className = sprintf('%05d', i) ;
  names = dir(fullfile(prefix, className, '*.ppm')) ;
  names = {names.name} ;
  for j = 1:numel(names)
    imagePath = fullfile(prefix, className, names{j}) ;
    t = imread(imagePath) ;
    t = imresize(im2single(t), [64 64]) ;
    im{end+1} = t ;
    lab(end+1) = i+1 ;
  end
end
im = cat(4, im{:}) ;
trainImages = im ;
trainLabels = lab ;

save('data/signs-train.mat', 'trainImages', 'trainLables') ;







pause