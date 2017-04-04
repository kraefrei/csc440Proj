clear; close all; clc;
%% Preliminary Data Conditioning
% the data file train.csv was first conditioned to remove all spaces 
% allowing for tableread to properly load the data.
%
if ~exist('training_data.mat')
    tab = readtable('train.csv');
    % This section corrects for numeric data being loaded as a string
    
    for i = 1:width(tab)
        if ~isnumeric(table2array(tab(:,i)))
            t_test_samp = table2array(tab(:,i));
            for j = 1:length(t_test_samp);
                t_test(j,1) = ~isempty(str2num(t_test_samp{j,1}));

            end
            if sum(t_test) > 0;
                t_cell_fill = cell(1,1);
                t_cell_fill{1,1} = '0';
                t_test_samp(~t_test) = t_cell_fill;
                t_right_tab = tab(:,(i+1:end));
                t_left_tab = tab(:,1:i);
                label = t_left_tab.Properties.VariableNames{i};
                t_left_tab(:,i) = []; % Delete column
                t_left_tab(:,i) = array2table(cellfun(@str2num, t_test_samp));
                t_left_tab.Properties.VariableNames{i} = label;
                tab = [t_left_tab t_right_tab];
            end
        end
    end
    clear t_* i j
    save training_data;
else
    load( 'training_data.mat' );
end
objects = size(tab,1);
% Find the unique labels for each attribute ignoring ID and final cost
datalabels = cell(width(tab)-2,1);
for i = 2:width(tab)-1
    datalabels{i-1} = unique(table2array(tab(:,i)));
    strlabels(i) = iscell(datalabels{i-1});
end
% Separating Numeric Atributes from catagorical
str_attributes = tab(:,strlabels);
num_attributes = tab(:,~strlabels);

% Finding missing values and calculating percent missing
for i = 1:height(str_attributes)
    for j = 1:width(str_attributes)
        tmp(i,j) = strcmp( char(str_attributes{i,j}), 'NA');
    end
end
percent_missing_str = sum(tmp)./length(tmp);

% Calculating missing numerical values (assuming both NaN and 0 are 
% possible missing entries) 
for i = 1:height(num_attributes)
    for j = 1:width(num_attributes)
        tmp1(i,j) = isnan(num_attributes{i,j});
        tmp2(i,j) = num_attributes{i,j} == 0;
    end
end
clear tmp;

% Filtering for attributes that realistically have 0 as a velid entry
% (This is a coarse check, but it seems to work)
sel = max(table2array(num_attributes)) < 10;
tmp = (tmp1 | tmp2);
tmp = tmp & ~repmat(sel, size(tmp1,1), 1);

percent_missing_num = sum(tmp)./length(tmp);

% Assumption that if there is more than 20% missing data, the attribute is
% not worth observing (We might want to look at some combined features here
% instead of throwing all these away?)
num_att_trim = num_attributes(:,(percent_missing_num < .15));
str_att_trim = str_attributes(:,(percent_missing_str < .15));

% Generating fresh unique string labels from new catagorical table
str_att_trim_arr = table2array(str_att_trim);
str_att_num = num_att_trim(:,1);
repl = cell(1);
for i = 1:size(str_att_trim,2)
    catagories{i,1} = unique(str_att_trim_arr(:,i));
    if size(catagories{i},1) == 2 % Let dual catagories be binary attrib.
        str_att_num(:,end+1) = table(strcmp(str_att_trim{:,i}, catagories{i}(1))); % store attribute in table
        str_att_num.Properties.VariableNames{end} = ...
            str_att_trim.Properties.VariableNames{i};
%     elseif size(catagories{i},1) == 3 % Let triple cat be trinary 
%         repl{1} = 1;
%         str_att_trim{strcmp(str_att_trim{:,i}, catagories{i}(1)),i} = repl;
%         repl{1} = 0;
%         str_att_trim{strcmp(str_att_trim{:,i}, catagories{i}(2)),i} = repl;
%         repl{1} = -1;
%         str_att_trim{strcmp(str_att_trim{:,i}, catagories{i}(3)),i} = repl;
%         str_att_num(:,end+1) = str_att_trim(:,i); % store attribute in table
    else % let each catagory be a binary catagory
        tmp = cell(1,size(catagories{i},1));
        for j = 1:size(catagories{i},1)
            tmp{j} = strcmp(str_att_trim{:,i}, catagories{i}(j));
            str_att_num(:,end+1) = table(tmp{j});
            name = str_att_trim.Properties.VariableNames{i};
            str_att_num.Properties.VariableNames{end} = [name '_', catagories{i}{j}];
        end
    end
end
%%
refined_mat = [table2array(str_att_num) table2array(num_att_trim) tab.SalePrice];
refined_tab = [str_att_num, num_att_trim(:,2:end), tab(:,end)];
refined_mat(:,end) = arrayfun(@log, refined_mat(:,end));

refined_mat = (refined_mat(:,2:end) - repmat( mean(refined_mat(:,2:end)),objects,1))...
                ./repmat(std(refined_mat(:,2:end)),objects,1);
            


%% Regressive SVM
% Model with conditioned data, binary catagories
%for C = 1:50
C = 10;
model = fitrsvm(refined_mat(1:1200,2:end-1), refined_mat(1:1200,end),'BoxConstraint',C*.1);
cvmodel = model.crossval();
for i = 1:10
    norm_solution(i,:) = cvmodel.Trained{i}.predict(refined_mat(1201:end,2:end-1));
end

plot(mean(norm_solution),'.')
hold on
plot(refined_mat(1201:end,end),'.');

%% For undoing solution normalization
mean_saleprice = mean(log(tab.SalePrice));
std_saleprice  = std(log(tab.SalePrice));

sol = mean(norm_solution).*std_saleprice' + mean_saleprice';
figure
grtrth = tab.SalePrice(1201:end);
test = exp(sol);
plot(test,'.')
hold on
plot(grtrth,'.')
rating = sqrt(mean((log(test'+1) - log(grtrth + 1)).^2));

%end

% rmsle(sol,refined_mat(1201:end,end))
% Normalization & Naive attempt
% TODO: Condition string values with integers and breaking up larger
% catagories into multiple binary catagories

