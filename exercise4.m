% EXERCISE4
setup ;

% Training cofiguration
%targetClass = 1 ;
%targetClass = 'prohibitory' ;
targetClass = 'mandatory' ;
%targetClass = 'danger' ;
numHardNegativeMiningIterations = 7 ;
schedule = [1 2 5 5 100 100 100] ;

% Scale space configuration
hogCellSize = 8 ;
minScale = -1 ;
maxScale = 3 ;
numOctaveSubdivisions = 3 ;
scales = 2.^linspace(...
  minScale,...
  maxScale,...
  numOctaveSubdivisions*(maxScale-minScale+1)) ;

% Load data
loadData(targetClass) ;

% Compute HOG features of examples (see Step 1.2)
trainBoxHog = {} ;
for i = 1:size(trainBoxPatches,4)
  trainBoxHog{i} = vl_hog(trainBoxPatches(:,:,:,i), hogCellSize) ;
end
trainBoxHog = cat(4, trainBoxHog{:}) ;
modelWidth = size(trainBoxHog,2) ;
modelHeight = size(trainBoxHog,1) ;

% -------------------------------------------------------------------------
% Step 4.1: Train with hard negative mining
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
  figure(1) ; clf ;
  imagesc(vl_hog('render', w)) ;
  colormap gray ; axis equal ;
  title(sprintf('SVM HOG model (retraining ieration %d)',t)) ;

  % Evaluate on training data and mine hard negatives
  figure(2) ; set(gcf, 'name', sprintf('Retraining iteration %d',t)) ;
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
% Step 4.2: Evaluate the model on the test data
% -------------------------------------------------------------------------

figure(3) ; clf ;
evaluateModel(...
    testImages, testBoxes, testBoxImages, ...
    w, hogCellSize, scales) ;