clear all
close all

disp('Importing data for analysis...');
A = importdata('letter-recognition.data');
[numrows,~] = size(A.data);

split_res = .1;    % resolution of cross-validation data
range_max = floor(1/split_res);

i = 4000;
train_data = A.data(1:i,:);
train_class = A.textdata(2:i+1,1);
test_data = A.data(i+1:end,:);
test_class = A.textdata(i+2:end,1);

c = cvpartition(train_class,'kfold',10); % stratified cv folds
stree = fitctree(train_data,train_class,'cvpartition',c);
num_nodes = zeros(stree.KFold,1);

for i = 1:stree.KFold
    m = numel(stree.Trained{i}.IsBranchNode);
    num_nodes(i,1) = m-nnz(stree.Trained{i}.IsBranchNode);% num leaf nodes
end
[avg_nodes,~] = max(num_nodes); % average number of leaves

trainspace = linspace(floor(split_res*avg_nodes),avg_nodes,range_max);
trainspace(end) = [];
n = numel(trainspace);
test_err = zeros(n,2);
train_err = zeros(n,2);
avg_err = zeros(n,2);
idx = 1;
for i = floor(trainspace)
    stree = fitctree(train_data,train_class,'CrossVal','on',...
        'MinLeafSize',i,'MinParentSize',2*i);
%     c = cvpartition(train_class,'kfold',10); % stratified cv folds
%     stree = fitctree(train_data,train_class,'cvpartition',c,...
%         'MinLeafSize',i,'MinParentSize',2*i);
    L = kfoldLoss(stree,'mode','individual'); % calculate training error
    avg_err(idx,1) = mean(L); % average training error
    [train_err(idx,1),smidx] = min(L); % index of best classifier model

    % use best classifier on test data
    test_err(idx,1) = loss(stree.Trained{smidx},test_data,test_class);
    
    idx = idx+1;
end

figure % learning curves
plot(trainspace,train_err(:,1),'-r',trainspace,avg_err(:,1),'-b',...
    trainspace,test_err(:,1),'-g');
title('classification tree learning curves')
xlabel('tree depth (branching nodes)')
ylabel('error')
legend('training (min)','training (avg)','test','location','best')

% for i = trainspace
% % 10-fold cv tree
% tree = fitctree(train_data,train_class,'CrossVal','on','MaxNumSplits',i);
% L = kfoldLoss(tree,'mode','individual'); % calculate training error
% avg_err(idx,1) = mean(L); % average training error
% [train_err(idx,1),midx] = min(L); % index of best classifier model
% 
% % use best classifier on test data
% test_err(idx,1) = loss(tree.Trained{midx},test_data,test_class);
% idx = idx+1;
% end