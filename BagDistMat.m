function distMat = BagDistMat(data,bagModels)

% distMat is an embedding of data which can be used to find fastBoot
% distances.
% Use pdist2 with cityblock distance to compute the fastboot distance
% between the embedded points

distMat = zeros(size(data,1),length(bagModels)*length(bagModels{1}),'single');
count = 1;
for mno = 1:length(bagModels)
  curModel = bagModels{mno};
  for j = 1:length(curModel)
    curWk = curModel(j);
    dd = data(:,curWk.dim)*curWk.dir;
    tt = curWk.tr*curWk.dir;
    distMat(:,count) = sign( (dd>tt)-0.5)*curWk.alpha;
    count = count+1;
  end
end
