function [Y,G,Q ] = fast_PDD_3L1( X_train,Q,I,Y,G,max_out,maxin,rho,lamda,ind,M,X_hat1 )
%ADMM algorithm for solving the regularized GCCA Problem
%     min 1/2?_i ?_j ||XiQi-XjQj||^2_F + ||Qi||_1
%     s.t.   Qi'Xi'XiQi=I

%    augmented Lagrangian min 1/2?_i ?_j ||XiQi-XjQj||^2_F + ||Qi||_1 + rho/2*|| XiQi-Gi+Ui||^2_F


%    input: X set of Xi matrices LxM
%           K number of canonical vectors
%           I number of datasets(views)
%   output: Q set of Qi matrices MxK
%           G=Xi*Qi;
%           U ADMM Dual variable
p=1:I; % set of datastets index
a=rho+I-1; % constant needed for reformulation
%% preprocessing for the first step
for i=1:I
    A=X_train{i};
    LL=svds(A,1);
    Lf(i)=LL^2;
end
clear A
for iter=1:max_out %outer iterations
    %     disp(['running at iteration',num2str(iter)]);
    %% BSUM ALGORITHM
    for in_iter=1:maxin
        % %%%%%step 1%%%%%%
        parfor i=1:I
            Kmat=rho*(G{i}-(1/rho)*Y{i}); % matrix needed for reformulation
            s=setdiff(p,i); % set of j~=i
            Gtot=0;
            for j=s
                Gtot=Gtot+G{j};
            end
            Mmat=(1/a)*(Gtot+Kmat);
%             clear Gtot Kmat
            lamda_tot=lamda/a;%lamda after reformulation
            [ Q{i},~ ] = FISTA_v1_L1( X_train{i},Q{i},Mmat,Lf(i),20,lamda_tot );
%             clear  Mmat
        end
        %% %%%%step 2%%%%%%%
        parfor i=1:I
            G_hat{i}=X_train{i}*Q{i};
        end
        clear temp
        parfor i=1:I
            Kmat=rho*(G_hat{i}+(1/rho)*Y{i});% matrix needed for reformulation
            s=setdiff(p,i);% set of j~=i
            XQtot=0;
            for j=s
                XQtot=XQtot+G_hat{j};
            end
            Mmat=(1/a)*(XQtot+Kmat);
%             clear XQtot Kmat
            [Um,~,Vm]=svd(Mmat,'econ');%solve procrustes problem using SVD
            G{i}=Um*Vm';
%             clear Um Vm Mmat
        end
        
        %%%BSUM stopping critirion
        
        if iter<2
            break
        end
    end
    %%%end BSUM
%         sum1=0;
% for i=1:I
%     Q_pddl21_hat1{i}=Q{i}(ind{i}(1:M),:);
%     Q_pddl21_hat2{i}=Q{i}(ind{i}(M+1:end),:);
%     sum1=sum1+norm(Q_pddl21_hat2{i},'fro');
%     row_sum{i}=sum(Q{i}.^2,2);
%     row_sum{i}=row_sum{i}(ind{i});
% end
% disp(['cost2: ',num2str(objXQ(I,X_hat1,Q_pddl21_hat1))]);
% disp(['COST2: ',num2str(sum1)]);
    %%    %%%%step 3%%%%%%%
    parfor i=1:I
        Y{i}=Y{i}+rho*(G_hat{i}-G{i});%dual variable update
    end
end
end
