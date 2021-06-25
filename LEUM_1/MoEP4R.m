tic; clear;clc;
addpath(genpath(pwd));
load('u.mat');
user_num = 944; item_num = 1683;
[rating_Ave, X_Noi] = preprocess_dataset(u, user_num, item_num);
m = user_num; n = item_num;
r = 4;
init_num = 5;               % the number of initialization
% PMoEP
CC_PMoEP = 0.015;           % tune between(0,0.3]
len_lam_PMoEP = length(CC_PMoEP);
Ind = randperm(m*n);
m_rate = 0.2;
p1 = floor(m*n*m_rate);
W = ones(m,n);
W(Ind(1:p1)) = 0;    
data_num = 1;
X_Ori = X_Noi;

for i = 1:init_num
    aa = median(abs(X_Noi(Ind(p1+1:end))));
    aa = sqrt(aa/r);
    U0 = rand(m,r)*aa*2-aa;
    V0 = rand(n,r)*aa*2-aa;
    
    for l = 1:len_lam_PMoEP
            
            % parameter setting 
            param_PMoEP.maxiter = 100;      % the maximum iteration number
            param_PMoEP.OriX = X_Noi;       % the clean data
            param_PMoEP.InU = U0;           % the initialization of U
            param_PMoEP.InV = V0;           % the initialization of V
            param_PMoEP.k = 5;              % the initialized number of mixture components
            param_PMoEP.display = 0;        % the display setting
            param_PMoEP.tol = 1.0e-10;      % the tolerance
            p = [2,1.8,1,0.5,0.1];          % can be tuned
            
            C= CC_PMoEP(l);
            disp(['lambda is: ',num2str(C)]);
            
            % call main function
            [label,model,TW,OutU,OutV,llh,llh_BIC,p] = EM_PMoEP(W,X_Noi,r,param_PMoEP,p,C);
            
            hat_K = length(model.Pi);
            N = numel(X_Noi)*(1-m_rate);
            D_k = 2;
  
            A = OutU; B = OutV;
            E1G_PMoEP(i,l) = sum(sum(abs(W.*(X_Noi-A*B'))));
            E2G_PMoEP(i,l) = sum(sum((W.*(X_Noi - A*B')).^2));
            E3G_PMoEP(i,l) = sum(sum(((X_Ori - A*B')).^2));
            E4G_PMoEP(i,l) = sum(sum(abs((X_Ori-A*B'))));
            % E5G_PMoEP(i,l,kk) = subspace(RU,A);
            % E6G_PMoEP(i,l,kk) = subspace(RV',B);
            E7G_PMoEP(i,l) = sum(sum(((X_Ori -  A*B')).^2))/sum(sum((X_Ori).^2));
            E8G_PMoEP(i,l) = sum(sum((W.*(X_Ori -  A*B')).^2))/sum(sum(( W.*X_Ori).^2));
            BIC_PMoEP(i,l) = llh_BIC - 0.5*2*hat_K*log(N);
            LLH_PMoEP(i,l) = llh(end);
            KK_PMoEP(i,l) = hat_K;
            P{i,l,kk} = p;
    end
end

for i = 1:init_num
    [value ind] = max(BIC_PMoEP(i,:));
    EE1G_PMoEP(i) = E1G_PMoEP(ind);
    EE2G_PMoEP(i) = E2G_PMoEP(i,ind);
    EE3G_PMoEP(i) = E3G_PMoEP(i,ind);
    EE4G_PMoEP(i) = E4G_PMoEP(i,ind);
    % EE5G_PMoEP(i) = E5G_PMoEP(i,ind);
    % EE6G_PMoEP(i) = E6G_PMoEP(i,ind);
    EE7G_PMoEP(i) = E7G_PMoEP(i,ind);
    EE8G_PMoEP(i) = E8G_PMoEP(i,ind);
    LLHH_PMoEP(i) = LLH_PMoEP(i,ind);
    KKK_PMoEP(i) = KK_PMoEP(i,ind);
    CCC_PMoEP(i) = CC_PMoEP(ind);
    PP{i} = P{i,ind};
end

[value,ii] = max(LLHH_PMoEP(:));
EEE1G_PMoEP = EE1G_PMoEP(ii);
EEE2G_PMoEP = EE2G_PMoEP(ii);
EEE3G_PMoEP = EE3G_PMoEP(ii);
EEE4G_PMoEP = EE4G_PMoEP(ii);
% EEE5G_PMoEP = EE5G_PMoEP(ii);
% EEE6G_PMoEP = EE6G_PMoEP(ii);
EEE7G_PMoEP = EE7G_PMoEP(ii);
EEE8G_PMoEP = EE8G_PMoEP(ii);
KKKK_PMoEP = KKK_PMoEP(ii);
CCCC_PMoEP = CCC_PMoEP(ii);
LLHHH_PMoEP = LLHH_PMoEP(ii);
PPP = PP{ii};
I = ii;

m_EEE1G_PMoEP = mean(EEE1G_PMoEP); m_EEE2G_PMoEP = mean(EEE2G_PMoEP); m_EEE3G_PMoEP = mean(EEE3G_PMoEP);
m_EEE4G_PMoEP = mean(EEE4G_PMoEP); % m_EEE5G_PMoEP = mean(EEE5G_PMoEP); m_EEE6G_PMoEP = mean(EEE6G_PMoEP);
m_EEE7G_PMoEP = mean(EEE7G_PMoEP); m_EEE8G_PMoEP = mean(EEE8G_PMoEP);


disp(['For the ',1,'_th data, ','the ',num2str(I),'_th initialization is selected. ','The number of mixture compoments is ',...
      num2str(KKKK_PMoEP),' and the selected p is ',num2str(PPP),'.'])


format short e;
M_Result_PMoEP = [m_EEE1G_PMoEP m_EEE2G_PMoEP m_EEE3G_PMoEP m_EEE4G_PMoEP...
                  m_EEE7G_PMoEP m_EEE8G_PMoEP]
disp(['Perforance of the eight measures are ',num2str(M_Result_PMoEP)]);
toc;            
              