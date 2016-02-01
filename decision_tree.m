% decision_tree.m assumes that the first column contains the classes

clear 

disp('Importing data for analysis...');
A = importdata('skillcraft_labeled.csv');
[numrows,numcols] = size(A.data);

split_res = .05;    % resolution of cross-validation data
range_max = floor(1/split_res);
trainspace = linspace(floor(split_res*numrows),numrows,range_max);
trainspace(end) = [];

[m,n] = size(trainspace);
test_err = zeros(n,1);
train_err = zeros(n,1);
avg_err = zeros(n,1);
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

    tree = fitrtree(train_data,train_class);
    cvtree = crossval(tree,'kfold',10); % 10-fold cross validation on tree
    L = kfoldLoss(cvtree,'mode','individual'); % calculate training error
    avg_err(idx,1) = mean(L); % average training error
    [train_err(idx,1),midx] = min(L); % index of best classifier model
    
    % use best classifier on test data
    test_err(idx,1) = loss(cvtree.Trained{midx},test_data,test_class);
    
    idx = idx+1;
end

plot(100*split_res*(1:size(train_err,1)),(train_err),'-r',...
    100*split_res*(1:size(test_err,1)),(test_err),'-g');
title('classification tree learning curves')
xlabel('training data (%)')
ylabel('error')
legend('training','test','location','best')
