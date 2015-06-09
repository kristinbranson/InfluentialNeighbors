%% Load the mnist data. The mat file can be downloaded from http://www.cs.nyu.edu/~roweis/data/mnist_all.mat

D = load('mnist_all.mat');

%% Train a d1 vs d2 boosting detector with decision stumps

d1 = 7;
d2 = 9;

% Edit d1 and d2 to run the demo on other pair of digits.

eval(sprintf('datad1 = double(D.train%d);',d1));
eval(sprintf('datad2 = double(D.train%d);',d2));
data = [datad1;datad2];
numd1 = size(datad1,1);
numd2 = size(datad2,1);
labels = [ones(numd1,1); zeros(numd2,1)];

% Preprocess the data
params = getDefaultParams;
binVals = findThresholds(data,params);
bins = findThresholdBins(data,binVals);

% Learn a boosting classifier
cl = boostingWrapper(data,labels,binVals,bins,params);

%% Measure accuracy.

eval(sprintf('testd1 = double(D.test%d);',d1));
eval(sprintf('testd2 = double(D.test%d);',d2));

predd1 = myBoostClassify(testd1,cl);
predd2 = myBoostClassify(testd2,cl);

errord1 = nnz(predd1<=0);
errord2 = nnz(predd2>0);

fprintf('Errors rate for %d vs %d: %.2f%%\n',d1,d2, (errord1+errord2)/(numel(predd1)+numel(predd2))*100);

%% Learn the fastboot model

fbcl = fastBoot(data,labels,binVals,bins,params);

%% Pick a mispredicted test examples 

mispred = find(predd1<=0);
sel = randsample(mispred,1);

%% Find the closest examples from both the classes for the selected test example

matd1 = BagDistMat(datad1,{fbcl});
matd2 = BagDistMat(datad2,{fbcl});
matsel = BagDistMat(testd1(sel,:),{fbcl});

dist_sel2d1 = pdist2(matd1,matsel,'CityBlock');
dist_sel2d2 = pdist2(matd2,matsel,'CityBlock');

[dist_sort_d1,ord_sort_d1] = sort(dist_sel2d1,'ascend');
[dist_sort_d2,ord_sort_d2] = sort(dist_sel2d2,'ascend');

%% Create a visualization of the classifier 

clIm = zeros(1,size(data,2));
for ndx = 1:numel(cl)
  if cl(ndx).dir<0
    clIm(cl(ndx).dim) = clIm(cl(ndx).dim)-cl(ndx).alpha;
  else
    clIm(cl(ndx).dim) = clIm(cl(ndx).dim)+cl(ndx).alpha;
  end
end
clIm = reshape(clIm,[28 28])';

%% Show the selected example and the closest 5 training examples

all_axis = [];
figure;
all_axis(end+1) = subplot(3,5,6); 
selIm = reshape(testd1(sel,:),[28 28])';
imshow(selIm/255);
title('Test image');

all_axis(end+1) = subplot(3,5,7);
imagesc(clIm); axis image; axis off;
title({'Feature weights',sprintf('Brighter - predict %d',d1),sprintf('Darker - predict %d',d2)} );

meand1 = reshape(mean(datad1,1),[28 28])';
meand2 = reshape(mean(datad2,1),[28 28])';

all_axis(end+1) = subplot(3,5,8);
imshow(meand1/255);
title(sprintf('Mean %d image',d1));
all_axis(end+1) = subplot(3,5,9);
imshow(meand2/255);
title(sprintf('Mean %d image',d2));


for ndx = 1:5
  all_axis(end+1) = subplot(3,5,ndx);
  d1Im = reshape(datad1(ord_sort_d1(ndx),:),[28 28])';
  imshow(d1Im/255);
  title(sprintf('dist:%.2f',dist_sort_d1(ndx)));
  all_axis(end+1) = subplot(3,5,10+ndx);
  d2Im = reshape(datad2(ord_sort_d2(ndx),:),[28 28])';
  imshow(d2Im/255);
  title(sprintf('dist:%.2f',dist_sort_d2(ndx)));
  
end

linkaxes(all_axis);
