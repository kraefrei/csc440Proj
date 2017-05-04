clear; close all; clc;
%% Preliminary Data Conditioning
% the data file train.csv was first conditioned to remove all spaces 
% allowing for tableread to properly load the data.

[str_att_num, num_att_trim, output] = process_data();
output = output(output > 0);
log_tform = 1;

monthdummy = array2table(dummyvar(num_att_trim{:,19}));
str_att_num = [str_att_num monthdummy];
lotarea2 = num_att_trim{:,3}.^2;
overallqual2 = num_att_trim{:,4}.^2;
overallcond2 = num_att_trim{:,5}.^2;
garagearea2 = num_att_trim{:,11}.^2;
addvars = [array2table(lotarea2) array2table(garagearea2) array2table(overallqual2) array2table(overallcond2)];
num_att_trim = [num_att_trim addvars];
%%
clear tmp
tmp = table2array(num_att_trim);
for i = 1:width(num_att_trim)
   tmp(:,i) = boxcox(table2array(num_att_trim(:,i))+0.1);
   tmp(:,i) = (tmp(:,i) - mean(tmp(:,i)))./std(tmp(:,i));
end

%refined_mat(:,end) = arrayfun(@log, refined_mat(:,end));
if log_tform
    norm_sale = (log(output) - mean(log(output))) ./std(log(output));
else
    norm_sale = (output - mean(output)) ./std(output);  
end

refined_mat = [table2array(str_att_num(1:1460,:)) tmp(1:1460,2:end), norm_sale];
refined_mat_test = [table2array(str_att_num(1461:end,:)) tmp(1461:end,2:end)];
all_data = [refined_mat(:,1:end-1);refined_mat_test];

%% Perfomring Random Holdout
% random_train = randperm(1460,1460);
% 
% Xnn = refined_mat(random_train(1:1200),2:end-1);
% Ynn = refined_mat(random_train(1:1200),end);

%% Clustering and PCA
%scatter of transformed data
clustered = 0;
N = 2;
close all
dim = 20;
pca_mat = pca(all_data(:,2:end));
twodtrans = all_data(:,2:end)*pca_mat(:,1:dim);
twodtrans = [twodtrans(1:1460,:) output];

toDelete = find(twodtrans(:,1)>6);
numOutliers = sum(twodtrans(:,1)>6);
refined_mat(toDelete,:) = [];
twodtrans(toDelete,:) = [];
clear cluster cluster_points out
close all
out_tmp = kmeans(twodtrans(:,1:dim),N,'distance','correlation','replicates',10,'start','sample');
cluster = zeros(1460-numOutliers,2);
out = zeros(1460-numOutliers,N);
for i = 1:N
    out(:,i) = (out_tmp(1:1460-numOutliers) == i);
    cluster_points = logical(out(1:(1460-numOutliers),i));
    cluster(cluster_points,1:3) = twodtrans(cluster_points,[1:2,end]);
    scatter3(cluster(cluster_points,1),cluster(cluster_points,2),log(cluster(cluster_points,end)))
    hold on
end
out = logical(out);
toDelete2 = find(twodtrans(:,2)>6);
numOutliers2 = sum(twodtrans(:,2)>6);
refined_mat(toDelete2,:) = [];
twodtrans(toDelete2,:) = [];

output1 = output;
output1(toDelete,:) = [];
output1(toDelete2,:) = [];
grtrth = output1;%(1201:end);

%%
% % %Add Interactions
% inter1 = refined_mat(:,279).*refined_mat(:,283);
% inter2 = refined_mat(:,279).*refined_mat(:,284);
% inter3 = refined_mat(:,279).*refined_mat(:,285);
% inter4 = refined_mat(:,284).*refined_mat(:,285);
% 
% refined_mat = [refined_mat(:,1:end-1) inter1 inter2 inter3 inter4 refined_mat(:,end)];
% 
% inter1T = refined_mat_test(:,279).*refined_mat_test(:,283);
% inter2T = refined_mat_test(:,279).*refined_mat_test(:,284);
% inter3T = refined_mat_test(:,279).*refined_mat_test(:,285);
% inter4T = refined_mat_test(:,284).*refined_mat_test(:,285);
% 
% refined_mat_test = [refined_mat_test inter1T inter2T inter3T inter4T];

Xnn = refined_mat(1:1200,2:end-1);
Ynn = refined_mat(1:1200,end);

%%FEATURESELECTION%%%%%%%%
%fun = @(Xtrain,Ytrain,Xtest,Ytest) sqrt(mean((log(predict((fitrgp(Xtrain,Ytrain)),Xtest)) - log(Ytest)).^2));
%options = statset('MaxIter',10);
%[fs,history] = sequentialfs(fun,refined_mat(:,2:end-1),refined_mat(:,end),'options',options);
%% Lasso Fitting
[lasso_holout,fitInfo1] = lasso(Xnn,Ynn,'CV',5);
[lasso_test,fitInfo2] = lasso(refined_mat(:,2:end-1),refined_mat(:,end),'CV',5);
%Delete columns where lasscoefficient is 0
toDeleteLasso = find(lasso_test(:,fitInfo2.Index1SE)==0);
for i=1:size(all_data,2)-1
    if (abs(pca_mat(i,1)))>0.1
        tempRemove = find(toDeleteLasso==i);
        toDeleteLasso(tempRemove,:)=[];
    end
end
toDeleteLasso((find(toDeleteLasso>222)),:)=[];
Xnn(:,toDeleteLasso) = [];
toDeleteLasso_rf = toDeleteLasso+1;
refined_mat(:,toDeleteLasso_rf)=[];
refined_mat_test(:,toDeleteLasso_rf)=[];
[lasso_holout2,fitInfo12] = lasso(Xnn,Ynn,'CV',5);
[lasso_test2,fitInfo22] = lasso(refined_mat(:,2:end-1),refined_mat(:,end),'CV',5);

holdOutData = refined_mat(1201:end,2:end-1);
%% Gaussian Process Model
% Model with conditioned data, binary catagories
folds = 10;
test_solutionG = zeros(size(refined_mat_test,1),1);
if ~clustered
gprMDL   = fitrgp(refined_mat(:,2:end-1),refined_mat(:,end),'KFold',folds,'Sigma',.274804); 
norm_solutionG = kfoldPredict(gprMDL);
for i = 1:folds
    test_solutionG = test_solutionG + predict(gprMDL.Trained{i}, refined_mat_test(:,2:end))./folds;
end
else
    test_solutionG = zeros(1459,1);
    for i = 1:N
       gpmdl_train{i} = fitrgp(refined_mat(out(1:1200,i),2:end-1),refined_mat(out(1:1200,i),end));
       gpmdl_test{i} = fitrgp(refined_mat(out(:,i),2:end-1),refined_mat(out(:,i),end));
       norm_solutionG(out(1201:end,i),1) = predict(gpmdl_train{i}, refined_mat(out(1201:end,i),2:end-1));
       %test_solutionG(out(1461:end,i),1) = predict(gpmdl_test{i}, refined_mat_test(out(1461:end,i),2:end));
    end
end
%% Ridge Fitting
kt = 0:.00005:.005;
ridge_holout = ridge(Ynn,Xnn,kt,0);
ridge_test = ridge(refined_mat(:,end),refined_mat(:,2:end-1),kt,0);
%%
norm_solutionL = (holdOutData)*(lasso_holout2(:,fitInfo12.Index1SE));
test_solutionL = (refined_mat_test(:,2:end))*(lasso_test2(:,fitInfo22.Index1SE));

colselect = 1;
norm_solutionR = (holdOutData)*(ridge_holout(2:end,colselect))+(ridge_holout(1,colselect));
test_solutionR = (refined_mat_test(:,2:end))*(ridge_test(2:end,colselect))+(ridge_test(1,colselect));
%% For undoing solution normalization

if log_tform
    %log
    mean_saleprice = mean(log(output));
    std_saleprice  = std(log(output));
else
    %notlog
    mean_saleprice = mean(output);
    std_saleprice  = std(output);
end

solG = norm_solutionG.*std_saleprice' + mean_saleprice';
solG_test = test_solutionG.*std_saleprice' + mean_saleprice';

solL = norm_solutionL.*std_saleprice' + mean_saleprice';
solL_test = test_solutionL.*std_saleprice' + mean_saleprice';

solR = norm_solutionR.*std_saleprice' + mean_saleprice';
solR_test = test_solutionR.*std_saleprice' + mean_saleprice';

if ~log_tform
    %not log
    testG = solG;
    final = solG_test;
    
    testL = solL;
    finalL = solL_test;
    
     testR = solR;
     finalR = solR_test;
else
    %log
    testG = exp(solG);
    final = exp(solG_test);
    
    testL = exp(solL);
    finalL = exp(solL_test);
    
     testR = exp(solR);
     finalR = exp(solR_test);
end
figure
% 
 finalRG = (finalR+final)/2;
% finalRL = (finalR+finalL)/2;
% finalLG = (final+finalL)/2;
% finalRLG = (finalR+finalL+final)/3;
%testRG = (testR+testG)/2;
% testRL = (testR+testL)/2;
% %testLG = (testG+testL)/2;
% %testRLG = (testR+testG+testL)/3;
% %plot(testG,'.')
%hold on
%plot(grtrth,'.')
ratingG = sqrt(mean((log(testG+1) - log(grtrth + 1)).^2));
%ratingL = sqrt(mean((log(testL+1) - log(grtrth + 1)).^2));
%ratingR = sqrt(mean((log(testR+1) - log(grtrth + 1)).^2));
%ratingRG = sqrt(mean((log(testRG+1) - log(grtrth + 1)).^2));
%ratingRL = sqrt(mean((log(testRL+1) - log(grtrth + 1)).^2));
%ratingLG = sqrt(mean((log(testLG+1) - log(grtrth + 1)).^2));
%ratingRLG = sqrt(mean((log(testRLG+1) - log(grtrth + 1)).^2));
%% Test Output to file
out_matG = [refined_mat_test(:,1) final];
dlmwrite('reg_solutionG.csv',out_matG,'precision',20);

% out_matR = [refined_mat_test(:,1) finalR];
% dlmwrite('reg_solutionR.csv',out_matR,'precision',20);
% 
% out_matL = [refined_mat_test(:,1) finalL];
% dlmwrite('reg_solutionL.csv',out_matL,'precision',20);
% 
 out_matRG = [refined_mat_test(:,1) finalRG];
 dlmwrite('reg_solutionRG.csv',out_matRG,'precision',20);
% 
% out_matRL = [refined_mat_test(:,1) finalRL];
% dlmwrite('reg_solutionRL.csv',out_matRL,'precision',20);
% 
% out_matLG = [refined_mat_test(:,1) finalLG];
% dlmwrite('reg_solutionLG.csv',out_matLG,'precision',20);
% 
% out_matRLG = [refined_mat_test(:,1) finalRLG];
% dlmwrite('reg_solutionRLG.csv',out_matRLG,'precision',20);