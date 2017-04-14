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
   %tmp(:,i) = boxcox(table2array(num_att_trim(:,i))+1 - min(table2array(num_att_trim(:,i))));
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
Xnn = refined_mat(1:1200,2:end-1);
Ynn = refined_mat(1:1200,end);

%% Clustering and PCA
%scatter of transformed data
clustered = 0;
N = 3;
close all
dim = 20;
pca_mat = pca(all_data(:,2:end-1));
twodtrans = all_data(:,2:end-1)*pca_mat(:,1:dim);
%scatter3(twodtrans(:,1),twodtrans(:,2),out)
out = kmeans(twodtrans(:,1:dim),N,'distance','correlation','replicates',10,'start','sample');
%end
out1 = (out == 1);
out2 = (out == 2);
out3 = (out == 3);
out = [out1, out2, out3];
%load clusters.mat;
twodtrans = [twodtrans(1:1460,:) output];
cluster1 = twodtrans(out1(1:1460),:);
cluster2 = twodtrans(out2(1:1460),:);
cluster3 = twodtrans(out3(1:1460),:);
scatter3(cluster1(:,1),cluster1(:,2),log(cluster1(:,end)))
hold on
scatter3(cluster2(:,1),cluster2(:,2),log(cluster2(:,end)))
scatter3(cluster3(:,1),cluster3(:,2),log(cluster3(:,end)))

%% Gaussian Process Model
% Model with conditioned data, binary catagories
if ~clustered
gprMDL_holout = fitrgp(Xnn,Ynn);
gprMDL_test   = fitrgp(refined_mat(:,2:end-1),refined_mat(:,end)); 
norm_solutionG = predict(gprMDL_holout, refined_mat(1201:1460,2:end-1));
test_solutionG = predict(gprMDL_test, refined_mat_test(:,2:end));
else
    test_solutionG = zeros(1459,1);
    for i = 1:N
       gpmdl_train{i} = fitrgp(refined_mat(out(1:1300,i),2:end-1),refined_mat(out(1:1300,i),end));
       gpmdl_test{i} = fitrgp(refined_mat(out(1:1460,i),2:end-1),refined_mat(out(1:1460,i),end));
       test_solutionG(out(1461:end,i),1) = predict(gpmdl{i}, refined_mat_test(out(1461:end,i),2:end));
    end
end
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
if ~log_tform
    %not log
    testG = solG;
    final = solG_test;
else
    %log
    testG = exp(solG);
    final = exp(solG_test);
end
figure
grtrth = output(1201:1460);

plot(testG,'.')
hold on
plot(grtrth,'.')
ratingG = sqrt(mean((log(testG+1) - log(grtrth + 1)).^2));


%% Test Output to file
out_mat = [refined_mat_test(:,1) final];
dlmwrite('reg_solution.csv',out_mat,'precision',20);

%end

% rmsle(sol,refined_mat(1201:end,end))
% Normalization & Naive attempt
% TODO: Condition string values with integers and breaking up larger
% catagories into multiple binary catagories