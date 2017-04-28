clear; close all; clc;
%% Preliminary Data Conditioning
% the data file train.csv was first conditioned to remove all spaces 
% allowing for tableread to properly load the data.

[str_att_num, num_att_trim, output] = process_data();
output = output(output > 0);
log_tform = 1;
%%
clear tmp
tmp = table2array(num_att_trim);
for i = 1:width(num_att_trim)
   tmp(:,i) = boxcox(table2array(num_att_trim(:,i))+1 - min(table2array(num_att_trim(:,i))));
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
grtrth = output1(1201:end);

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
holdOutData = refined_mat(1201:end,2:end-1);

%%FEATURESELECTION%%%%%%%%
%fun = @(Xtrain,Ytrain,Xtest,Ytest) sqrt(mean((log(predict((fitrgp(Xtrain,Ytrain)),Xtest)) - log(Ytest)).^2));
%options = statset('MaxIter',10);
%[fs,history] = sequentialfs(fun,refined_mat(:,2:end-1),refined_mat(:,end),'options',options);
%% Gaussian Process Model
% Model with conditioned data, binary catagories
if ~clustered
gprMDL_holout = fitrgp(Xnn,Ynn);
gprMDL_test   = fitrgp(refined_mat(:,2:end-1),refined_mat(:,end)); 
norm_solutionG = predict(gprMDL_holout, holdOutData);
test_solutionG = predict(gprMDL_test, refined_mat_test(:,2:end));

else
    test_solutionG = zeros(1459,1);
    for i = 1:N
       gpmdl_train{i} = fitrgp(refined_mat(out(1:1200,i),2:end-1),refined_mat(out(1:1200,i),end));
       gpmdl_test{i} = fitrgp(refined_mat(out(:,i),2:end-1),refined_mat(out(:,i),end));
       norm_solutionG(out(1201:end,i),1) = predict(gpmdl_train{i}, refined_mat(out(1201:end,i),2:end-1));
       %test_solutionG(out(1461:end,i),1) = predict(gpmdl_test{i}, refined_mat_test(out(1461:end,i),2:end));
    end
end

%% Lasso Fitting
[lasso_holout,fitInfo1] = lasso(Xnn,Ynn,'CV',10);
[lasso_test,fitInfo2] = lasso(refined_mat(:,2:end-1),refined_mat(:,end),'CV',10);
%% Ridge Fitting
kt = 0:1e-5:5e-3;
ridge_holout = ridge(Ynn,Xnn,kt,0);
ridge_test = ridge(refined_mat(:,end),refined_mat(:,2:end-1),kt,0);
%%
norm_solutionL = (holdOutData)*(lasso_holout(:,fitInfo1.Index1SE));
test_solutionL = (refined_mat_test(:,2:end))*(lasso_test(:,fitInfo2.Index1SE));

colselect = 500;
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

finalAVG = (finalR+final+finalL)/3;
testAVG = (testR+testG+testL)/3;
plot(testG,'.')
hold on
plot(grtrth,'.')
ratingG = sqrt(mean((log(testG+1) - log(grtrth + 1)).^2));
ratingL = sqrt(mean((log(testL+1) - log(grtrth + 1)).^2));
ratingAVG = sqrt(mean((log(testAVG+1) - log(grtrth + 1)).^2));
ratingR = sqrt(mean((log(testR+1) - log(grtrth + 1)).^2));
%% Test Output to file
out_mat = [refined_mat_test(:,1) final];
dlmwrite('reg_solution.csv',out_mat,'precision',20);

out_matAVG = [refined_mat_test(:,1) finalAVG];
dlmwrite('reg_solutionAVG.csv',out_matAVG,'precision',20);
