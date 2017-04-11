clear; close all; clc;
%% Preliminary Data Conditioning
% the data file train.csv was first conditioned to remove all spaces 
% allowing for tableread to properly load the data.

[str_att_num, num_att_trim, output] = process_data();
output = output(output > 0);
%%
clear tmp
tmp = table2array(num_att_trim);
for i = 1:width(num_att_trim)
   %tmp(:,i) = boxcox(table2array(num_att_trim(:,i))+1 - min(table2array(num_att_trim(:,i))));
   tmp(:,i) = (tmp(:,i) - mean(tmp(:,i)))./std(tmp(:,i));
end

%refined_mat(:,end) = arrayfun(@log, refined_mat(:,end));

norm_sale = (output - mean(output)) ./std(output);
refined_mat = [table2array(str_att_num(1:1460,:)) tmp(1:1460,2:end), norm_sale];
refined_mat_test = [table2array(str_att_num(1461:end,:)) tmp(1461:end,2:end)];

%refined_tab = [str_att_num, num_att_trim(:,2:end), tab(:,end)];

%refined_mat(:,[248 252 253 260]) = arrayfun(@log, refined_mat(:,[248 251 252 253 260]));

Xnn = refined_mat(1:1200,2:end-1);
Ynn = refined_mat(1:1200,end);
            


%% Regressive SVM
% Model with conditioned data, binary catagories
%for C = 1:50

C = .011;
model = fitrsvm(refined_mat(1:1200,2:end-1), refined_mat(1:1200,end),'BoxConstraint',C);
norm_solutionS = model.predict(refined_mat(1201:1460,2:end-1));
% cvmodel = model.crossval();
% for i = 1:10
%     norm_solution(i,:) = cvmodel.Trained{i}.predict(refined_mat(1201:1460,2:end-1));
% end
t = templateTree('NumPredictorsToSample','all',...
    'PredictorSelection','interaction-curvature','Surrogate','on');
ens = fitensemble(refined_mat(1:1200,2:end-1),refined_mat(1:1200,end),'bag',100,'Tree','Type','regression');
norm_solutionE = predict(ens, refined_mat(1201:1460,2:end-1));

gprMDL = fitrgp(Xnn,Ynn);
norm_solutionG = predict(gprMDL, refined_mat(1201:1460,2:end-1));
test_solutionG = predict(gprMDL, refined_mat_test(:,2:end));

plot(norm_solutionE,'.')
hold on
plot(refined_mat(1201:1460,end),'.');

%% For undoing solution normalization
%scatter of transformed data
close all
pca_mat = pca(refined_mat(:,2:end-1));
twodtrans = refined_mat(:,2:end-1)*pca_mat(:,1:2);
%scatter3(twodtrans(:,1),twodtrans(:,2),out)
twodtrans = [twodtrans output];
out = kmeans(twodtrans(:,1:2),2,'distance','cosine','replicates',10,'start','sample');
%end
out1 = (out == 1);
out2 = (out == 2);
%out3 = (out == 3);
load clusters.mat;
cluster1 = twodtrans(out,:);
cluster2 = twodtrans(~out,:);
%cluster3 = twodtrans(out3,:);
scatter3(cluster1(:,1),cluster1(:,2),log(cluster1(:,3)))
hold on
scatter3(cluster2(:,1),cluster2(:,2),log(cluster2(:,3)))
%scatter3(cluster3(:,1),cluster3(:,2),cluster3(:,3))
%log
%mean_saleprice = mean(log(num_att_trim.SalePrice));
%std_saleprice  = std(log(num_att_trim.SalePrice));

%notlog
mean_saleprice = mean(output);
std_saleprice  = std(output);

solE = norm_solutionE.*std_saleprice' + mean_saleprice';
solS = norm_solutionS.*std_saleprice' + mean_saleprice';
solG = norm_solutionG.*std_saleprice' + mean_saleprice';
solG_test = test_solutionG.*std_saleprice' + mean_saleprice';

%not log
testE = solE;
testS = solS;
testG = solG;

%log
%testE = exp(solE);
%testS = exp(solS);

figure
grtrth = output(1201:1460);

plot(solG,'.')
hold on
plot(grtrth,'.')
ratingEns = sqrt(mean((log(testE+1) - log(grtrth + 1)).^2));
ratingSVM = sqrt(mean((log(testS+1) - log(grtrth + 1)).^2));
ratingG = sqrt(mean((log(testG+1) - log(grtrth + 1)).^2));


%% Test Output to file
out_mat = [refined_mat_test(:,1) solG_test];
dlmwrite('reg_solution.csv',out_mat,'precision',20);

%end

% rmsle(sol,refined_mat(1201:end,end))
% Normalization & Naive attempt
% TODO: Condition string values with integers and breaking up larger
% catagories into multiple binary catagories