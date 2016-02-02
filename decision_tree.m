% decision_tree.m assumes that the first column contains the classes

clear 

disp('Importing data for analysis...');
A = importdata('letter-recognition.data');
[numrows,numcols] = size(A.data);

split_res = .1;    % resolution of cross-validation data
range_max = floor(1/split_res);
trainspace = linspace(floor(split_res*numrows),numrows,range_max);
trainspace(end) = [];

[m,n] = size(trainspace);
test_err = zeros(n,2);
train_err = zeros(n,2);
avg_err = zeros(n,2);
idx = 1;
rng(42);    % set random seed

disp('Computing cross validation error...');
for i = trainspace
    prog = sprintf('Run %d/%d...',idx,range_max-1);
    disp(prog);
    
    train_data = A.data(1:i,:);
    train_class = A.textdata(2:i+1,1);
    test_data = A.data(i+1:end,:);
    test_class = A.textdata(i+2:end,1);

    % stratified decision tree
    c = cvpartition(train_class,'kfold',10); % stratified cv folds
    stree = fitctree(train_data,train_class,'cvpartition',c);
    L = kfoldLoss(stree,'mode','individual'); % calculate training error
    avg_err(idx,1) = mean(L); % average training error
    [train_err(idx,1),smidx] = min(L); % index of best classifier model
    
    % use best classifier on test data
    test_err(idx,1) = loss(stree.Trained{smidx},test_data,test_class);
    
%     % 10-fold cv tree
%     tree = fitctree(train_data,train_class);
%     cvtree = crossval(tree,'kfold',10); % 10-fold cross validation
%     L = kfoldLoss(cvtree,'mode','individual'); % calculate training error
%     avg_err(idx,2) = mean(L); % average training error
%     [train_err(idx,2),midx] = min(L); % index of best classifier model
%     
%     % use best classifier on test data
%     test_err(idx,2) = loss(cvtree.Trained{midx},test_data,test_class);
    
    idx = idx+1;
end

figure(1) % learning curves
plot(100*split_res*(1:size(train_err,1)),train_err(:,1),'-r',...
    100*split_res*(1:size(avg_err,1)),avg_err(:,1),'-b',...
    100*split_res*(1:size(test_err,1)),test_err(:,1),'-g');
title('classification tree learning curves')
xlabel('training data (%)')
ylabel('error')
legend('train (min)','train (avg)','test','location','best')

figure(2) % confusion matrix
[labels,scores] = kfoldPredict(stree);
cmat = confusionmat(stree.Y,labels);
heatmap(cmat,stree.Y,stree.Y,1,'Colormap','red','ShowAllTicks',true,...
    'UseLogColorMap',true,'Colorbar',true,'Gridlines',':')

