% EXERCISE5
setup ;

% Training cofiguration
targetClass = 1 ;
numHardNegativeMiningIterations = 5 ;
schedule = [1 2 5 5 5] ;

% Scale space configuration
hogCellSize = 8 ;
minScale = -1 ;
maxScale = 3 ;
numOctaveSubdivisions = 3 ;
scales = 2.^linspace(...
  minScale,...
  maxScale,...
  numOctaveSubdivisions*(maxScale-minScale+1)) ;

% -------------------------------------------------------------------------
% Step 5.1: Construct custom training data
% -------------------------------------------------------------------------

% Load object examples
trainImages = {} ;
trainBoxes = [] ;
trainBoxPatches = {} ;
trainBoxImages = {} ;
trainBoxLabels = [] ;

% Construct negative data
names = dir('data/myNegatives/*.jpeg') ;
trainImages = fullfile('data', 'myNegatives', {names.name}) ;

% Construct positive data
names = dir('data/myPositives/*.jpeg') ;
names = fullfile('data', 'myPositives', {names.name}) ;
for i=1:numel(names)
  im = imread(names{i}) ;
  im = imresize(im, [64 64]) ;
  trainBoxes(:,i) = [0.5 ; 0.5 ; 64.5 ; 64.5] ;
  trainBoxPatches{i} = im2single(im) ;
  trainBoxImages{i} = names{i} ;
  trainBoxLabels(i) = 1 ;
end
trainBoxPatches = cat(4, trainBoxPatches{:}) ;

% Compute HOG features of examples (see Step 1.2)
trainBoxHog = {} ;
for i = 1:size(trainBoxPatches,4)
  trainBoxHog{i} = vl_hog(trainBoxPatches(:,:,:,i), hogCellSize) ;
end
trainBoxHog = cat(4, trainBoxHog{:}) ;
modelWidth = size(trainBoxHog,2) ;
modelHeight = size(trainBoxHog,1) ;

% -------------------------------------------------------------------------
% Step 5.2: Visualize the training images
% -------------------------------------------------------------------------

figure(1) ; clf ;

subplot(1,2,1) ;
imagesc(vl_imarraysc(trainBoxPatches)) ;
axis off ;
title('Training images (positive samples)') ;
axis equal ;

subplot(1,2,2) ;
imagesc(mean(trainBoxPatches,4)) ;
box off ;
title('Average') ;
axis equal ;

% -------------------------------------------------------------------------
% Step 5.3: Train with hard negative mining
% -------------------------------------------------------------------------

% Initial positive and negative data
pos = trainBoxHog(:,:,:,ismember(trainBoxLabels,targetClass)) ;
neg = zeros(size(pos,1),size(pos,2),size(pos,3),0) ;

for t=1:numHardNegativeMiningIterations
  numPos = size(pos,4) ;
  numNeg = size(neg,4) ;
  C = 1 ;
  lambda = 1 / (C * (numPos + numNeg)) ;
  
  fprintf('Hard negative mining iteration %d: pos %d, neg %d\n', ...
    t, numPos, numNeg) ;
    
  % Train an SVM model (see Step 2.2)
  x = cat(4, pos, neg) ;
  x = reshape(x, [], numPos + numNeg) ;
  y = [ones(1, size(pos,4)) -ones(1, size(neg,4))] ;
  w = vl_svmtrain(x,y,lambda,'epsilon',0.01,'verbose') ;
  w = single(reshape(w, modelHeight, modelWidth, [])) ;

  % Plot model
  figure(2) ; clf ;
  imagesc(vl_hog('render', w)) ;
  colormap gray ;
  axis equal ;
  title('SVM HOG model') ;
  
  % Evaluate on training data and mine hard negatives
  figure(3) ;  
  [matches, moreNeg] = ...
    evaluateModel(...
    vl_colsubset(trainImages', schedule(t), 'beginning'), ...
    trainBoxes, trainBoxImages, ...
    w, hogCellSize, scales) ;
  
  % Add negatives
  neg = cat(4, neg, moreNeg) ;
  
  % Remove negative duplicates
  z = reshape(neg, [], size(neg,4)) ;
  [~,keep] = unique(z','stable','rows') ;
  neg = neg(:,:,:,keep) ;  
end


% -------------------------------------------------------------------------
% Step 5.3: Evaluate the model on the test data
% -------------------------------------------------------------------------

im = imread('data/myTestImage.jpeg') ;
im = im2single(im) ;

% Compute detections
[detections, scores] = detect(im, w, hogCellSize, scales) ;
keep = boxsuppress(detections, scores, 0.25) ;
detections = detections(:, keep(1:10)) ;
scores = scores(keep(1:10)) ;

% Plot top detection
figure(3) ; clf ;
imagesc(im) ; axis equal ;
hold on ;
vl_plotbox(detections, 'g', 'linewidth', 2, ...
  'label', arrayfun(@(x)sprintf('%.2f',x),scores,'uniformoutput',0)) ;
title('Multiple detections') ;