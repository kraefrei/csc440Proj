function [str_att_num, num_att_trim, out, att] = process_data( )
var{1} = 'train';
var{2} = 'test';
for v = 1:2
    if ~exist([var{1},'_data','.mat'])
        tab = readtable(str);
        % This section corrects for numeric data being loaded as a string
        for i = 1:width(tab)
            if ~isnumeric(table2array(tab(:,i)))
                if i<width(tab)
                    t_test_sampX = table2array(tab(:,i));
                end
                t_test_samp = table2array(tab(:,i));
                for j = 1:length(t_test_samp)
                    t_test(j,1) = ~isempty(str2num(t_test_samp{j,1}));
                    if j<length(t_test_sampX)
                        t_testX(j,1) = ~isempty(str2num(t_test_sampX{j,1}));
                    end
                end
                if sum(t_test) > 0
                    t_cell_fill = cell(1,1);
                    t_cell_fill{1,1} = 'NaN';
                    t_test_samp(~t_test) = t_cell_fill;
                    t_right_tab = tab(:,(i+1:end));
                    t_left_tab = tab(:,1:i);
                    label = t_left_tab.Properties.VariableNames{i};
                    t_left_tab(:,i) = []; % Delete column
                    t_left_tab(:,i) = array2table(cellfun(@str2num, t_test_samp));
                    t_left_tab.Properties.VariableNames{i} = label;
                    tab = [t_left_tab t_right_tab];
                end
                if (sum(t_testX) > 0) && (i<width(tab))
                    t_cell_fill = cell(1,1);
                    t_cell_fill{1,1} = 'NaN';
                    t_test_sampX(~t_testX) = t_cell_fill;
                    t_right_tab = tab(:,(i+1:end));
                    t_left_tab = tab(:,1:i);
                    label = t_left_tab.Properties.VariableNames{i};
                    t_left_tab(:,i) = []; % Delete column
                    t_left_tab(:,i) = array2table(cellfun(@str2num, t_test_sampX));
                    t_left_tab.Properties.VariableNames{i} = label;
                    tab = [t_left_tab t_right_tab];
                end
            end
        end
        t_tmp = cell(size(tab,1),1);
        for i = 1:size(tab,1)   
            t_tmp{i,1} = num2str(tab{i,2});
        end

        t_name = tab.Properties.VariableNames{2};
        tab(:,2) = [];
        t_y = tab(:,end);
        tab(:,end) = [];
        tab{:,end+1} = t_tmp;
        tab.Properties.VariableNames{end} = t_name;
        tab = [tab t_y];
        clear t_* i j str label
        save([var{1},'_data']);
    end
end
load('train_data.mat')
train_tab = tab;
clear tab
load('test_data.mat')
test_tab = tab;
tmp = test_tab(:,end-1);
test_tab(:,end-1) = [];
test_tab = [test_tab, tmp];
test_tab(:,end+1) = table(zeros(height(tab),1));
test_tab.Properties.VariableNames{end} = train_tab.Properties.VariableNames{end};

tab = [train_tab;test_tab];
clear tmp;

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
str_test = tab(:,strlabels);
num_test = tab(:,~strlabels);

% Finding missing values and calculating percent missing
for i = 1:height(str_attributes)
    for j = 1:width(str_attributes)
        tmp(i,j) = strcmp( char(str_attributes{i,j}), 'NA');
    end
end
percent_missing_str = sum(tmp)./length(tmp);
clear tmp;
% selecting attributes that use 0 as a missing value
sel = zeros(1,width(num_attributes));
sel(1,[6,7,25,36]) = 1;
% Calculating missing numerical values (assuming both NaN and 0 are 
% possible missing entries) 
for i = 1:height(num_attributes)
    for j = 1:width(num_attributes)
        if sel(j) == 1 && num_attributes{i,j} == 0
            num_attributes{i,j} = NaN;
        end
        tmp(i,j) = isnan(num_attributes{i,j}) || num_attributes{i,j} == 0;
    end
end

percent_missing_num = sum(tmp)./length(tmp);

% Assumption that if there is more than 20% missing data, the attribute is
% not worth observing (We might want to look at some combined features here
% instead of throwing all these away?)
num_att_trim = num_attributes(:,(percent_missing_num < .20));
att{1} = percent_missing_num < .20;
str_att_trim = str_attributes(:,(percent_missing_str < .20));
att{2} = percent_missing_str < .20;

% KNNinpute for filling missing data
num_att_trim_arr = table2array(num_att_trim);
num_att_trim_arr = knnimpute(num_att_trim_arr',5,'Distance','mahalanobis');
num_att_trim{:,:} = num_att_trim_arr';

% Generating fresh unique string labels from new catagorical table
%str_att_trim_arr = table2array(str_att_trim);
str_att_num = num_att_trim(:,1);
repl = cell(1);
ordered = zeros(1,size(str_att_trim,2)); % indicator for ordered catagories;
ordered(1,[18,19,21,22,23,27,30,34,35]) = 1;
for i = 1:size(str_att_trim,2)
    catagories{i,1} = unique(str_att_trim{:,i});
    if size(catagories{i},1) == 2 % Let dual catagories be binary attrib.
        str_att_num(:,end+1) = table(strcmp(str_att_trim{:,i}, catagories{i}(1))); % store attribute in table
        str_att_num.Properties.VariableNames{end} = ...
            str_att_trim.Properties.VariableNames{i};
            
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

out = tab.SalePrice;


end