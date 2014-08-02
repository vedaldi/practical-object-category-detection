
% -------------------------------------------------------------------------
% Step 1.0: Load training data
% -------------------------------------------------------------------------

setup ;

% Load the training and testing data (trainImages, trainBoxes, ...)
% The functio takes the ID of the type of traffic sign we want to recognize
% 1 is the 30 km/h speed limit
loadData(1) ;

% -------------------------------------------------------------------------
% Step 1.1: Visualize the training images
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
% Step 1.2: Extract HOG features from the training images
% -------------------------------------------------------------------------

hogCellSize = 8 ;
trainHog = {} ;
for i = 1:size(trainBoxPatches,4)
  trainHog{i} = vl_hog(trainBoxPatches(:,:,:,i), hogCellSize) ;
end
trainHog = cat(4, trainHog{:}) ;

% -------------------------------------------------------------------------
% Step 1.3: Learn a simple HOG template model
% -------------------------------------------------------------------------

w = mean(trainHog, 4) ;

save('data/signs-model-1.mat', 'w') ;

figure(2) ; clf ;
imagesc(vl_hog('render', w)) ;
colormap gray ;
axis equal ;
title('HOG model') ;

% -------------------------------------------------------------------------
% Step 1.4: Apply the model to a test image
% -------------------------------------------------------------------------

im = imread('data/signs-sample-image.jpg') ;
im = im2single(im) ;
hog = vl_hog(im, hogCellSize) ;
scores = vl_nnconv(hog, w, []) ;

figure(3) ; clf ;
imagesc(scores) ;
title('Detection') ;
colorbar ;

% -------------------------------------------------------------------------
% Step 1.5: Extract the top detection
% -------------------------------------------------------------------------

[best, bestIndex] = max(scores(:)) ;

[hy, hx] = ind2sub(size(scores), bestIndex) ;
x = (hx - 1) * hogCellSize + 1 ;
y = (hy - 1) * hogCellSize + 1 ;

modelWidth = size(trainHog, 2) ;
modelHeight = size(trainHog, 1) ;
detection = [
  x - 0.5 ;
  y - 0.5 ;
  x + hogCellSize * modelWidth - 0.5 ;
  y + hogCellSize * modelHeight - 0.5 ;] ;

figure(4) ; clf ;
imagesc(im) ; axis equal ;
hold on ;
vl_plotbox(detection, 'g', 'linewidth', 5) ;
title('Response scores') ;



