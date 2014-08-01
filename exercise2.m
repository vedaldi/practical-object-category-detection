
setup ;

load('data/signs-train.mat', ...
  'trainImages', ...
  'trainPatches', ...
  'trainLabels', ...
  'trainBoxes', ...
  'trainPatches', ...
  'testImages', ...
  'testLabels', ...
  'testBoxes', ...
  'testPatches') ;
hogCellSize = 6 ;
targetClass = 1 ;

% Compute HOG features of examples (see Step 1.2)
trainHog = {} ;
for i = 1:size(trainPatches,4)
  trainHog{i} = vl_hog(trainPatches(:,:,:,i), hogCellSize) ;
end
trainHog = cat(4, trainHog{:}) ;

% Learn a trivial HOG model (see Step 1.3)
w = mean(trainHog(:,:,:,trainLabels == targetClass), 4) ;
save('data/signs-model-1.mat', 'w') ;

figure(2) ; clf ;
imagesc(vl_hog('render', w)) ;
colormap gray ;
axis equal ;
title('Trivial HOG model') ;

% -------------------------------------------------------------------------
% Step 2.1: Multi-scale detection
% -------------------------------------------------------------------------

im = imread(testImages{16}) ;
im = im2single(im) ;

minScale = -1 ;
maxScale = 3 ;
numOctaveSubdivisions = 3 ;
scales = 2.^linspace(...
  minScale,...
  maxScale,...
  numOctaveSubdivisions*(maxScale-minScale+1)) ;

figure(5) ; clf ;
detection = detectAtMultipleScales(im, w, hogCellSize, scales) ;

figure(6) ; clf ;
imagesc(im) ; axis equal ;
hold on ;
vl_plotbox(detection, 'g', 'linewidth', 5) ;
title('Trivial detector output') ;

% -------------------------------------------------------------------------
% Step 2.2: Improving the model with an SVM
% -------------------------------------------------------------------------

% Collect positive training data
pos = trainHog(:,:,:,trainLabels == targetClass) ;

% Collect negative training data
neg = {} ;
modelWidth = size(trainHog, 2) ;
modelHeight = size(trainHog, 1) ;
for t=1:10
  % Get the HOG features of a training image
  t = imread(trainImages{16}) ;
  t = im2single(t) ;  
  hog = vl_hog(t, hogCellSize) ;
  
  % Sample uniformly 100 HOG patches
  % Assume that these are negative (almost certain)
  width = size(hog,2) - modelWidth + 1 ;
  height = size(hog,1) - modelHeight + 1 ;
  index = vl_colsubset(1:width*height, 100, 'uniform') ;

  for j=1:numel(index)
    [hy, hx] = ind2sub([height width], index(j)) ;
    sx = hx + (0:modelWidth-1) ;
    sy = hy + (0:modelHeight-1) ;
    neg{end+1} = hog(sy, sx, :) ;
  end
end
neg = cat(4, neg{:}) ;

numPos = size(pos,4) ;
numNeg = size(neg,4) ;
lambda = 0.01 ;

% Pack the data into a matrix with one datum per column
x = cat(4, pos, neg) ;
x = reshape(x, [], numPos + numNeg) ;

% Create a vector of binary labels
y = [ones(1, size(pos,4)) -ones(1, size(neg,4))] ;

% Learn the SVM using an SVM solver
w = vl_svmtrain(x,y,lambda,'epsilon',0.01,'verbose') ;

% Reshape the model vector into a model HOG template
w = single(reshape(w, modelHeight, modelWidth, [])) ;

% Plot model
figure(7) ; clf ;
imagesc(vl_hog('render', w)) ;
colormap gray ;
axis equal ;
title('SVM HOG model') ;

% Compute detections
figure(8) ; clf ;
detection = detectAtMultipleScales(im, w, hogCellSize, scales) ;

% Plot top detection
figure(9) ; clf ;
imagesc(im) ; axis equal ;
hold on ;
vl_plotbox(detection, 'g', 'linewidth', 5) ;
title('SVM detector output') ;

save('data/signs-model-2.mat', 'w') ;


