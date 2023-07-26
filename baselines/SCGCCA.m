% reproduction of  SCGCCA  by lkx
clc; clear; close all;
dataset = 'document';
if strcmp(dataset, 'synthetic')
%     [train_data,test_data, all_data] = generate_synthetic();
    train_data = load('syntrain_data.mat');
    train_data = {train_data.view0, train_data.view1,train_data.view2};
    test_data = load('syntest_data.mat');
    test_data = {test_data.view0, test_data.view1,test_data.view2};
elseif strcmp(dataset, 'genedata')
    train_data = load('Leukemia_train.mat');
    train_data = {train_data.view0, double(train_data.view1)};
    test_data = load('Leukemia_test.mat');
    test_data = {test_data.view0, double(test_data.view1)};
%     length = size(train_data,2);
elseif  strcmp(dataset, 'document')
%     [train_data,test_data, all_data] = generate_synthetic();
    train_data = load('document_train.mat');
    train_data = {train_data.view0, train_data.view1,train_data.view2};
    test_data = load('document_test.mat');
    test_data = {test_data.view0, test_data.view1,test_data.view2};
end
rng(0);
%%
%  for synthetic/document data
size_train_data = {size(train_data{1}) ,size(train_data{2}),size(train_data{3})};
s = 1; %r
samp = {size_train_data{1}(1),size_train_data{2}(1),size_train_data{3}(1)};  %samples
fea = {size_train_data{1}(2),size_train_data{2}(2),size_train_data{3}(2)};%features
P = {randn(fea{1},s), randn(fea{2},s), randn(fea{3},s)};%randn(1000, 1);
%% for Leukemia
% size_train_data = {size(train_data{1}) ,size(train_data{2})};
% s =1;
% samp = {size_train_data{1}(1),size_train_data{2}(1)};
% fea = {size_train_data{1}(2),size_train_data{2}(2)};
% P = {randn(fea{1},s), randn(fea{2},s)};
%% initailization:
tic;
U = randn(samp{1},s);
cnt = 0; %counter
time = 0;
%% SCGCCA
while 1
    cnt = cnt + 1;
    %% updating U
    temp = 0;
    for i = 1:numel(P)
        temp = temp + train_data{i}*P{i};
    end
    [Q, S, V] = svd(temp); 
    U = Q(:,1:s)*V';
    %% updating P via NHTP
    pars.s  = s;
%     pars.eta = 10000;
    pars.disp = 1;
    for i = 1:numel(P)
        data.A = train_data{i};
        data.b = U;
        out{i} = CSsolver(data, fea{i},'NHTP',pars);
        P{i} = out{i}.sol;
    end
    % results output and recovery display 
%     fprintf(' CPU time:     %.3fsec\n',out{1}.time+out{2}.time+out{3}.time);
    time = time+ out{1}.time+out{2}.time+out{3}.time;
%     time = time+ out{1}.time+out{2}.time;
    if cnt == 1
%         old_obj = out{1}.obj + out{2}.obj;
        old_obj = out{1}.obj + out{2}.obj+ out{3}.obj;
    else
%         new_obj = out{1}.obj + out{2}.obj;
        new_obj = out{1}.obj + out{2}.obj+ out{3}.obj;
        fprintf('in iter %d, convergence error = %f\n', cnt, abs(new_obj-old_obj));
        if abs(new_obj-old_obj) < 1e-4
            break;
        else
            old_obj = new_obj;
        end
    end  
    if cnt == 1
        break;
    end
end
elapsedTime = toc;
disp(['运行时间：' num2str(elapsedTime) ' 秒']);
list_projection= {train_data{1}*P{1}, train_data{2}*P{2}, train_data{3}*P{3}};
list_projection_test= {test_data{1}*P{1}, test_data{2}*P{2}, test_data{3}*P{3}};
% list_projection= {train_data{1}*P{1}, train_data{2}*P{2}};
% list_projection_test= {test_data{1}*P{1}, test_data{2}*P{2}};
G = solve_g(train_data);
% res = cal_G_error(train_data, G,P);
% fprintf(' Sample size:  %dx%d\n', m,n);
if strcmp(dataset, 'synthetic')
    fprintf("!! reconstruction error of G in training of SCGCCA is: %f\n",  cal_G_error(train_data, G,P));
    fprintf("!! reconstruction error of G in testing of SCGCCA  is: %f\n",  cal_G_error(test_data, G,P));
% %         # sum of correlation: Pearson product-moment correlation coefficients, [-1,1], abs=1 is better
    fprintf("!! total correlation in training data of SCGCCA is: %f\n", sum(cal_correlation(list_projection)));
    fprintf("!! total correlation in testing data of SCGCCA is: %f\n", sum(cal_correlation(list_projection_test))); %# projection for test data
%         train_cor.append(np.sum(model.cal_correlation(model.list_projection)))
%         test_cor.append(np.sum(model.cal_correlation(model.transform(data.test_data))))
    res = cal_spare(P); %# save info
    fprintf("!! each view's sparsity of SCGCCA is:  %f\n", res);
    fprintf("!! averaged sqarsity of SCGCCA is: %f\n", mean(res));
elseif strcmp(dataset, 'genedata')
    fprintf("!! reconstruction error of G in training of SCGCCA is: %f\n",  cal_G_error(train_data, G,P));
%     fprintf("!! reconstruction error of G in testing of SCGCCA  is: %f\n",  cal_G_error(test_data, G,P));
    fprintf("!! total correlation in training data of SCGCCA is: %f\n", sum(cal_correlation(list_projection)));
    fprintf("!! total correlation in testing data of SCGCCA is: %f\n", sum(cal_correlation(list_projection_test))); %# projection for test data
    res = cal_spare(P); %# save info
    fprintf("!! each view's sparsity of SCGCCA is:  %f\n", res);
    fprintf("!! averaged sqarsity of SCGCCA is: %f\n", mean(res));
    fprintf("!! classification accuracy in training of SCGCCA is: %f\n", cal_acc(list_projection));
    fprintf("!! classification accuracy in testing of SCGCCA is: %f\n", cal_acc(list_projection_test));
elseif strcmp(dataset, 'document')
    fprintf("!! reconstruction error of G in training of SCGCCA is: %f\n",  cal_G_error(train_data, G,P));
%     fprintf("!! reconstruction error of G in testing of SCGCCA  is: %f\n",  cal_G_error(test_data, G,P));
    fprintf("!! total correlation in training data of SCGCCA is: %f\n", sum(cal_correlation(list_projection)));
    fprintf("!! total correlation in testing data of SCGCCA is: %f\n", sum(cal_correlation(list_projection_test))); %# projection for test data
    res = cal_spare(P); %# save info
    fprintf("!! each view's sparsity of SCGCCA is:  %f\n", res);
    fprintf("!! averaged sqarsity of SCGCCA is: %f\n", mean(res));
    fprintf("!! AROC of each view in training of SCGCCA is: %f\n", cal_AROC(list_projection));
    fprintf("!! averaged AROC in training of SCGCCA is: %f\n", mean(cal_AROC(list_projection)));
    fprintf("!! AROC of each view in testing of SCGCCA is: %f\n", cal_AROC(list_projection_test));
    fprintf("!! averaged of each view in testing of SCGCCA is: %f\n", mean(cal_AROC(list_projection_test)));
end