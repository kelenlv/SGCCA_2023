function [ X,L ] = FISTA_v1_L1( A,X,B,Lf,Maxiter,lamda )
% min 1/2||AX-B||^2_F+lamda||X||_1 
X_hat=X;
L=Lf;theta=1;
const=A'*B;
for iter=1:Maxiter
    X_prev=X;
    Y=(1-theta)*X_prev+theta*X_hat;
    %% backtracking step
    L_prev=L;
    L=0.9*L;
    AY=A*Y;
    gradY=A'*AY-const;
    AgradY=A*gradY;
    costY=0.5*norm(AY-B,'fro')^2;
    for i=1:100
        X=Y-1/L*gradY;
        %%% soft thresholding operator
        temp1=abs(X)-(lamda*1/L);
        temp2=subplus(temp1);
        X=sign(X).*temp2;
        costX=0.5*norm(AY-1/L*AgradY-B,'fro')^2;
        clear temp1 temp2
        %% stoping criterion
        if norm(X-X_prev,'fro')/max(1,norm(X,'fro'))<1e-08
            break
        end        
        %% compute backtracking critirion
        temp3=(X-Y);
        temp4=gradY.*temp3;
        temp5=sum(sum(temp4));
        backtrack_crit=costX-costY-temp5-0.5*L*norm(temp3)^2;
        clear temp3 temp4 temp5
        if backtrack_crit>0
            L=2*L;
        else
            break
        end
    end
    X_hat=(X-(1-theta)*X_prev)/theta;
    theta=2/(1+sqrt(1+4*L/(L_prev*theta^2)));
end
end