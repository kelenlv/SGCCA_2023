clc;
clear all;
close all;
rng(0);
addpath cg_matlab
train_data = load('syntrain_data.mat');
train_data = {train_data.view0, train_data.view1,train_data.view2};
test_data = load('syntest_data.mat');
test_data = {test_data.view0, test_data.view1,test_data.view2};
size_train_data = {size(train_data{1}) ,size(train_data{2}),size(train_data{3})};
% s =1;
samp = {size_train_data{1}(1),size_train_data{2}(1),size_train_data{3}(1)};
fea = {size_train_data{1}(2),size_train_data{2}(2),size_train_data{3}(2)};
tic;
%% initailization:
I = 3; %views
K = 1; %number of canonical variables
P = {randn(fea{1},K), randn(fea{2},K), randn(fea{3},K)};
for i=1:I
%     [L,MM] = size(X{i});
%     G{i}=train_data{i}*P{i};
%     U = randn(samp{1},s);
    G{i}=randn(samp{i},K); % random initialization
%     [G{i},~,~] = svd(G{i},'econ');
    U{i}=sprandn(samp{i},K,1e-4); %sparsity_level = 1e-4
%     Q{i}=sprandn(M,K,sparsity_level); 
end
%% PDD parallel
out_iter=1000;
rho=0.2;
in_iter = 4; % select in_iter=1 for ADMM
% disp('operating parallel PDD with CG iteration');
% [U_final_pdd_par,G_pdd,Q_pdd ] = fast_PDD_noreg_par2(train_data,P,I,U,G,out_iter,in_iter,K,rho );
%% parallel PDD L1
lamda=1;
disp('running parallel PDD-l1');
[~,G_pddl1,Q_pddl1 ] = fast_PDD_3L1( train_data,P,I,U,G,out_iter,in_iter,rho,lamda );

% disp('running parallel PDD-l21');
% [U_pddl21,G_pddl21,Q_pddl21 ] = fast_PDD_3L21( train_data,P,I,U,G,out_iter,in_iter,rho,lamda);
% disp(['cost: ',num2str(costXQ(I,X,Q_pddl21))]);
elapsedTime = toc;
disp(['运行时间：' num2str(elapsedTime) ' 秒']);
list_projection= {train_data{1}*Q_pddl1{1}, train_data{2}*Q_pddl1{2}, train_data{3}*Q_pddl1{3}};
list_projection_test= {test_data{1}*Q_pddl1{1}, test_data{2}*Q_pddl1{2}, test_data{3}*Q_pddl1{3}};
% G = solve_g(train_data);
fprintf("!! reconstruction error of G in training of PDD is: %f\n",  cal_G_error(train_data, G,P));
fprintf("!! reconstruction error of G in testing of PDD  is: %f\n",  cal_G_error(test_data, G,P));
fprintf("!! total correlation in training data of PDD is: %f\n", sum(cal_correlation(list_projection)));
fprintf("!! total correlation in testing data of PDD is: %f\n", sum(cal_correlation(list_projection_test))); %# projection for test data
res = cal_spare(Q_pddl1); %# save info
fprintf("!! each view's sparsity of PDD is:  %f\n", res);
fprintf("!! averaged sqarsity of PDD is: %f\n", mean(res));
% fprintf("!! classification accuracy in training of PDD is: %f\n", cal_AROC(list_projection));
% fprintf("!! classification accuracy in testing of PDD is: %f\n", cal_AROC(list_projection_test));

