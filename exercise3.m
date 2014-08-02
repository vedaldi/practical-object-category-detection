% EXERCISE3
setup ;

% Feature configuration
hogCellSize = 8 ;
numHardNegativeMiningIterations = 3 ;
minScale = -1 ;
maxScale = 3 ;
numOctaveSubdivisions = 3 ;
scales = 2.^linspace(...
  minScale,...
  maxScale,...
  numOctaveSubdivisions*(maxScale-minScale+1)) ;

% Load data
load('data/signs-model-2.mat','w','targetClass') ;
loadData(targetClass) ;

% -------------------------------------------------------------------------
% Step 3.1: Multiple detections
% -------------------------------------------------------------------------

im = imread(testImages{3}) ;
im = im2single(im) ;

% Compute detections
[detections, scores] = detect(im, w, hogCellSize, scales) ;

% Non-maxima suppression
keep = boxsuppress(detections, scores, 0.25) ;

detections = detections(:, keep) ;
scores = scores(keep) ;

% Further keep only top detections
detections = detections(:, 1:10) ;
scores = scores(1:10) ;

% Plot top detection
figure(10) ; clf ;
imagesc(im) ; axis equal ;
hold on ;
vl_plotbox(detections, 'g', 'linewidth', 2, ...
  'label', arrayfun(@(x)sprintf('%.2f',x),scores,'uniformoutput',0)) ;
title('Multiple detections') ;

% -------------------------------------------------------------------------
% Step 3.2: Detector evaluation
% -------------------------------------------------------------------------

% Find all the objects in the target image
s = find(strcmp(testImages{3}, testBoxImages)) ;
gtBoxes = testBoxes(:, s) ;

% No example is considered difficult
gtDifficult = false(1, numel(s)) ;

% PASCAL-like evaluation
matches = evalDetections(...
  gtBoxes, gtDifficult, ...
  detections, scores) ;

% Visualization
figure(1) ; clf ;
imagesc(im) ; axis equal ; hold on ;
vl_plotbox(detections(:, matches.detBoxFlags==+1), 'g', 'linewidth', 2) ;
vl_plotbox(detections(:, matches.detBoxFlags==-1), 'r', 'linewidth', 2) ;
vl_plotbox(gtBoxes, 'b', 'linewidth', 1) ;
axis off ;

figure(2) ; clf ;
vl_pr(matches.labels, matches.scores) ;

% -------------------------------------------------------------------------
% Step 3.3: Evaluation on multiple images
% -------------------------------------------------------------------------

figure(3) ; clf ;

matches = evaluateModel(testImages, testBoxes, testBoxImages, ...
  w, hogCellSize, scales) ;

