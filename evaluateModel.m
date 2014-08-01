function [matches, negs] = evaluateModel(...
  testImages, testBoxes, testBoxImages, w, hogCellSize, scales)

clear matches ;
negs = {} ;
for i=1:numel(testImages)
  % Detect on test image
  im = imread(testImages{i}) ;
  im = im2single(im) ;
  [detections, scores, hog] = detect(im, w, hogCellSize, scales) ;

  % Non-maxima suppression
  keep = boxsuppress(detections, scores, 0.5) ;
  keep = find(keep) ;
  keep = vl_colsubset(keep, 30, 'beginning') ;
  detections = detections(:, keep) ;
  scores = scores(keep) ;

  % Find all the objects in the target image
  ok = find(strcmp(testImages{i}, testBoxImages)) ;
  gtBoxes = testBoxes(:, ok) ;
  gtDifficult = false(1, numel(ok)) ;
  matches(i) = evalDetections(...
    gtBoxes, gtDifficult, ...
    detections, scores) ;
  
  % Visualize progres
  clf;
  subplot(1,2,1) ;
  imagesc(im) ; axis equal ; hold on ;
  vl_plotbox(detections(:, matches(i).detBoxFlags==-1), 'r', 'linewidth', 1) ;
  vl_plotbox(detections(:, matches(i).detBoxFlags==+1), 'g', 'linewidth', 2) ;
  vl_plotbox(gtBoxes, 'b', 'linewidth', 1) ;
  axis off ;

  subplot(1,2,2) ;
  vl_pr([matches.labels], [matches.scores]) ;
  
  % If required, collect top negative features
  if nargout > 1
    overlaps = boxoverlap(gtBoxes, detections) ;
    overlaps(end+1,:) = 0 ;
    overlaps = max(overlaps,[],1) ;
    detections(:, overlaps >= 0.25) = [] ;
    detections = vl_colsubset(detections, 10, 'beginning') ;
    negs{end+1} = extract(hog, hogCellSize, scales, w, detections) ;
  end
  
  drawnow ;
end

if nargout > 1
  negs = cat(4, negs{:}) ;
end