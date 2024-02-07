% Specify the path to your JSON file
jsonFilePath = 'D:\Development\ffd-gnn\models\gridsearch_results_12_12_23.json';
jsonFilePath = 'D:\Development\ffd-gnn\models\gridsearch_results.json';

% Read the JSON file
jsonData = fileread(jsonFilePath);

% Decode the JSON data into a MATLAB structure
data = jsondecode(jsonData);

%%

gin = [data.x8K_5_v2.GIN.mean_f1, data.x16K_5_v2.GIN.mean_f1, data.x32K_5_v2.GIN.mean_f1, data.x64K_5_v2.GIN.mean_f1, data.x128K_5_v2_2.GIN.mean_f1];
sage = [data.x8K_5_v2.SAGE.mean_f1, data.x16K_5_v2.SAGE.mean_f1, data.x32K_5_v2.SAGE.mean_f1, data.x64K_5_v2.SAGE.mean_f1, data.x128K_5_v2_2.SAGE.mean_f1];
rf = [data.x8K_5_v2.RF.test_f1, data.x16K_5_v2.RF.test_f1, data.x32K_5_v2.RF.test_f1, data.x64K_5_v2.RF.test_f1, data.x128K_5_v2_2.RF.test_f1];
xg = [data.x8K_5_v2.XG.test_f1, data.x16K_5_v2.XG.test_f1, data.x32K_5_v2.XG.test_f1, data.x64K_5_v2.XG.test_f1, data.x128K_5_v2_2.XG.test_f1];
ginsage = [data.x8K_5_v2.GINSAGE.mean_f1, data.x16K_5_v2.GINSAGE.mean_f1, data.x32K_5_v2.GINSAGE.mean_f1, data.x64K_5_v2.GINSAGE.mean_f1, data.x128K_5_v2_2.GINSAGE.mean_f1];
%ginsage = [nan .82 .82 .80];
xn = 1:5;
figure;
hold on;    
plot(xn,gin,'-o');
plot(xn,sage,'-o');
plot(xn,ginsage,'-o')
plot(xn,rf,'-o');
plot(xn,xg,'-o');
xlim([0.5,5.5])

legend({'GIN','SAGE','GINSAGE','RF','XG'})

legend box off;

grid on;
xticks(xn)
xticklabels({'8k','16k','32k','64k','128k'})
title('5% fraudulent')


%%

%gin = [data.x128K_1_v2.GIN.test_f1, data.x128K_5_v2.GIN.test_f1, data.x128K_10_v2.GIN.test_f1];
sage = [data.x128K_1_v2.SAGE.test_f1, data.x128K_5_v2.SAGE.test_f1, data.x128K_10_v2.SAGE.test_f1];
%rf = [data.x128K_1_v2.RF.test_f1, data.x128K_5_v2.RF.test_f1, data.x128K_10_v2.RF.test_f1];
%xg = [data.x128K_1_v2.XG.test_f1, data.x128K_5_v2.XG.test_f1, data.x128K_10_v2.XG.test_f1];
ginsage = [data.x128K_1_v2.GINSAGE.test_f1, data.x128K_5_v2.GINSAGE.test_f1, data.x128K_10_v2.GINSAGE.test_f1];

xn = 1:3;
figure;
hold on;
%plot(xn,gin,'-o');
plot(xn,sage,'-o');
plot(xn,ginsage,'-o')
%plot(xn,rf,'-o');
%plot(xn,xg,'-o');

%legend({'GIN','SAGE','GINSAGE','RF','XG'})
legend({'SAGE','GINSAGE'})

legend box off;

grid on;
xticks(xn)
xticklabels({'128k 1%','128k 5%','128k 10%'})
title('5% fraudulent')

%%


%gin = [data.x128K_1_v2.GIN.test_f1, data.x128K_5_v2.GIN.test_f1, data.x128K_10_v2.GIN.test_f1];
sage = [data.x128K_05_v2.SAGE.mean_f1, data.x128K_1_v2.SAGE.mean_f1, data.x128K_5_v2.SAGE.mean_f1, data.x128K_10_v2.SAGE.mean_f1];
ginsage = [data.x128K_05_v2.GINSAGE.mean_f1, data.x128K_1_v2.GINSAGE.mean_f1, data.x128K_5_v2.GINSAGE.mean_f1, data.x128K_10_v2.GINSAGE.mean_f1];
xg = [data.x128K_05_v2.XG.test_f1, data.x128K_1_v2.XG.test_f1, data.x128K_5_v2.XG.test_f1, data.x128K_10_v2.XG.test_f1];
rf = [data.x128K_05_v2.RF.test_f1, data.x128K_1_v2.RF.test_f1, data.x128K_5_v2.RF.test_f1, data.x128K_10_v2.RF.test_f1];

xn = 1:4;
figure;
hold on;
%plot(xn,gin,'-o');
plot(xn,sage,'-o');
plot(xn,ginsage,'-o')
plot(xn,rf,'-o');
plot(xn,xg,'-o');

%legend({'GIN','SAGE','GINSAGE','RF','XG'})
legend({'SAGE','GINSAGE','XG','RF'})

legend box off;

grid on;
xticks(xn)
xticklabels({'128k 0.5%','128k 1%','128k 5%','128k 10%'})
title('scaling fraudulent percentages, fixed dataset size')


%%


gin = [data.x128K_05_v2_2.GIN.mean_f1, data.x128K_1_v2_2.GIN.mean_f1, data.x128K_5_v2_2.GIN.mean_f1, data.x128K_10_v2_2.GIN.mean_f1];
sage = [data.x128K_05_v2_2.SAGE.mean_f1, data.x128K_1_v2_2.SAGE.mean_f1, data.x128K_5_v2_2.SAGE.mean_f1, data.x128K_10_v2_2.SAGE.mean_f1];
ginsage = [data.x128K_05_v2_2.GINSAGE.mean_f1, data.x128K_1_v2_2.GINSAGE.mean_f1, data.x128K_5_v2_2.GINSAGE.mean_f1, data.x128K_10_v2_2.GINSAGE.mean_f1];
xg = [data.x128K_05_v2_2.XG.test_f1, data.x128K_1_v2_2.XG.test_f1, data.x128K_5_v2_2.XG.test_f1, data.x128K_10_v2_2.XG.test_f1];
rf = [data.x128K_05_v2_2.RF.test_f1, data.x128K_1_v2_2.RF.test_f1, data.x128K_5_v2_2.RF.test_f1, data.x128K_10_v2_2.RF.test_f1];

xn = 1:4;
figure;
hold on;
plot(xn,gin,'-o');
plot(xn,sage,'-o');
plot(xn,ginsage,'-o')
plot(xn,rf,'-o');
plot(xn,xg,'-o');

legend({'GIN','SAGE','GINSAGE','RF','XG'})
%legend({'SAGE','GINSAGE','XG','RF'})

legend box off;

grid on;
xticks(xn)
xticklabels({'128k 0.5%','128k 1%','128k 5%','128k 10%'})
title('scaling fraudulent percentages, fixed dataset size')

%%



fieldNames2 = fieldnames(data);


% Create a table with 6 columns and 3 rows
columnName = {'dataset', 'gin', 'sage', 'ginsage', 'xg', 'rf'};


fill_data = zeros(length(fieldNames2), 6);  % Random data for illustration

outTableMeanF1 = array2table(fill_data, 'VariableNames', columnName);
outTableStdF1 = array2table(fill_data, 'VariableNames', columnName);

outTableMeanF1.dataset = repmat("null", length(fieldNames2),1);
outTableStdF1.dataset = repmat("null", length(fieldNames2),1);

for j=1:numel(fieldNames2)
    fieldName2 = fieldNames2{j};
    ds = data.(fieldName2);
    
    outTableMeanF1.dataset(j) = fieldName2;
    outTableStdF1.dataset(j) = fieldName2;

    meanf1s = zeros(1,5);
    stdf1s = zeros(1,5);
    %ds = data.x8K_5_v2;
    fieldNames = fieldnames(ds);
    for i=1:numel(fieldNames)
        fieldName = fieldNames{i};
        fieldValue = ds.(fieldName);
    
        if ismember(fieldName,{'GIN','SAGE','GINSAGE'})
            if strcmp(fieldName,'GIN')
                idx = 1;
                outTableMeanF1.gin(j) = round(fieldValue.mean_f1, 5);
                outTableStdF1.gin(j) = (std(fieldValue.test_f1));
            elseif strcmp(fieldName,'SAGE')
                idx = 2;
                outTableMeanF1.sage(j) = round(fieldValue.mean_f1, 5);
                outTableStdF1.sage(j) = (std(fieldValue.test_f1));
            elseif strcmp(fieldName,'GINSAGE')
                idx = 3;
                outTableMeanF1.ginsage(j) = round(fieldValue.mean_f1, 5);
                outTableStdF1.ginsage(j) = (std(fieldValue.test_f1));
            end
            meanf1s(idx) = fieldValue.mean_f1;
            stdf1s(idx) = std(fieldValue.test_f1); 
        elseif ismember(fieldName,{'XG','RF'})
           if strcmp(fieldName,'XG')
                idx = 4;
                outTableMeanF1.xg(j) = round(fieldValue.test_f1, 5);
                outTableStdF1.xg(j) = 0.01;
            elseif strcmp(fieldName,'RF')
                idx = 5;
                outTableMeanF1.rf(j) = round(fieldValue.test_f1, 5);
                outTableStdF1.rf(j) = 0.01;
           end
           meanf1s(idx) = fieldValue.test_f1;
           stdf1s(idx) = 0.01; 
        else
               
        end
    end
end



%%

writetable(outTableMeanF1, 'dataset_models_results_mean.csv');
writetable(outTableStdF1, 'dataset_models_results_std.csv');
