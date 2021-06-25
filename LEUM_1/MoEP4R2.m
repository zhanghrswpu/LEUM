% Demo on Mixture Noise 1
tic; clear;clc;
addpath(genpath(pwd));
tic; clear;clc;
addpath(genpath(pwd));
load('u.mat');
user_num = 200; item_num = 400;
[rating_Ave, X_Noi, W, rating_counts] = preprocess_dataset(u, user_num, item_num);
rating_Ind = W;
disp("The data has been loaded! ");
m = user_num; n = item_num;  % data size    
r = 4;                      % rank
%X_Noi = sparse(X_Noi);
% m = 40; n = 20;             % data size
% r = 4;     
data_num = 1;               % the number of data
init_num = 5;               % the number of initialization

% PMoEP
CC_PMoEP = 0.005;           % tune between(0,0.3]
len_lam_PMoEP = length(CC_PMoEP);

for kk = 1:data_num
     disp(['Data ',num2str(kk)]);
     
     Ind = randperm(m*n);
     % m_rate = 0.2;
     % p1 = floor(m*n*m_rate);
     p1 = sum(sum(W~=0));
     m_rate = p1/(m*n);
     W = ones(m,n);
     W(Ind(1:p1)) = 0;       % Indicator matrix
     X_Ori = X_Noi;
     %W = sparse(W);
    for i = 1:init_num
        % random initialization for U0 and V0
        % aa = median(abs(X_Noi(1:end)));
        % aa = median(abs(X_Noi));
        aa = median(abs(X_Noi(Ind(1:end))));
        aa = sqrt(aa/r);
        U0 = rand(m,r)*aa*2-aa;
        V0 = rand(n,r)*aa*2-aa;
        
        for l = 1:len_lam_PMoEP
            
           % parameter setting 
            param_PMoEP.maxiter = 50;      % the maximum iteration number
            param_PMoEP.OriX = X_Noi;       % the clean data
            param_PMoEP.InU = U0;           % the initialization of U
            param_PMoEP.InV = V0;           % the initialization of V
            param_PMoEP.k = 5;              % the initialized number of mixture components
            param_PMoEP.display = 0;        % the display setting
            param_PMoEP.tol = 1.0e-4;      % the tolerance
            p = [2,1,0.8,0.5,0.1];          % can be tuned
            
            C= CC_PMoEP(l);
            disp(['lambda is: ',num2str(C)]);
            
            % call main function
            [label,model,TW,OutU,OutV,llh,llh_BIC,p] = EM_PMoEP(W,X_Noi,r,param_PMoEP,p,C);
            
            hat_K = length(model.Pi);
            N = numel(X_Noi)*(1-m_rate);
            D_k = 2;
            
            A = OutU; B = OutV;
            
            % calculate the measures
            % E1G_PMoEP(i,l,kk) = sum(sum(abs(W.*(X_Noi-A*B'))));
            % E2G_PMoEP(i,l,kk) = sum(sum((W.*(X_Noi - A*B')).^2));
            E3G_PMoEP(i,l,kk) = sum(sum(((X_Ori - A*B')).^2));
            E4G_PMoEP(i,l,kk) = sum(sum(abs((X_Ori-A*B'))));
            % E5G_PMoEP(i,l,kk) = subspace(RU,A);
            % E6G_PMoEP(i,l,kk) = subspace(RV',B);
            % E7G_PMoEP(i,l,kk) = sum(sum(((X_Ori -  A*B')).^2))/sum(sum((X_Ori).^2));
            % E8G_PMoEP(i,l,kk) = sum(sum((W.*(X_Ori -  A*B')).^2))/sum(sum(( W.*X_Ori).^2));
            BIC_PMoEP(i,l,kk) = llh_BIC - 0.5*2*hat_K*log(N);
            LLH_PMoEP(i,l,kk) = llh(end);
            KK_PMoEP(i,l,kk) = hat_K;
            P{i,l,kk} = p;
            % figure; plot(1:length(llh),llh); xlabel('Iteratin');ylabel('Objective Function Value')
        end
    end
end

for kk = 1:data_num
    for i = 1:init_num
        [value ind] = max(BIC_PMoEP(i,:,kk));
        % EE1G_PMoEP(kk,i) = E1G_PMoEP(i,ind,kk);
        % EE2G_PMoEP(kk,i) = E2G_PMoEP(i,ind,kk);
        EE3G_PMoEP(kk,i) = E3G_PMoEP(i,ind,kk);
        EE4G_PMoEP(kk,i) = E4G_PMoEP(i,ind,kk);
        % EE5G_PMoEP(kk,i) = E5G_PMoEP(i,ind,kk);
        % EE6G_PMoEP(kk,i) = E6G_PMoEP(i,ind,kk);
        % EE7G_PMoEP(kk,i) = E7G_PMoEP(i,ind,kk);
        % EE8G_PMoEP(kk,i) = E8G_PMoEP(i,ind,kk);
        LLHH_PMoEP(kk,i) = LLH_PMoEP(i,ind,kk);
        KKK_PMoEP(kk,i) = KK_PMoEP(i,ind,kk);
        CCC_PMoEP(kk,i) = CC_PMoEP(ind);
        PP{kk,i} = P{i,ind,kk};
    end
end

for kk = 1:data_num
    [value,ii] = max(LLHH_PMoEP(kk,:));
    % EEE1G_PMoEP(kk) = EE1G_PMoEP(kk,ii);
    % EEE2G_PMoEP(kk) = EE2G_PMoEP(kk,ii);
    EEE3G_PMoEP(kk) = EE3G_PMoEP(kk,ii);
    EEE4G_PMoEP(kk) = EE4G_PMoEP(kk,ii);
    % EEE5G_PMoEP(kk) = EE5G_PMoEP(kk,ii);
    % EEE6G_PMoEP(kk) = EE6G_PMoEP(kk,ii);
    % EEE7G_PMoEP(kk) = EE7G_PMoEP(kk,ii);
    % EEE8G_PMoEP(kk) = EE8G_PMoEP(kk,ii);
    KKKK_PMoEP(kk) = KKK_PMoEP(kk,ii);
    CCCC_PMoEP(kk) = CCC_PMoEP(kk,ii);
    % LLHHH_PMoEP(kk) = LLHH_PMoEP(kk,ii);
    PPP{kk} = PP{kk,ii};
    I(kk) = ii;
end
% m_EEE1G_PMoEP = mean(EEE1G_PMoEP); m_EEE2G_PMoEP = mean(EEE2G_PMoEP); m_EEE3G_PMoEP = mean(EEE3G_PMoEP);
% m_EEE4G_PMoEP = mean(EEE4G_PMoEP); m_EEE5G_PMoEP = mean(EEE5G_PMoEP); m_EEE6G_PMoEP = mean(EEE6G_PMoEP);
% m_EEE7G_PMoEP = mean(EEE7G_PMoEP); m_EEE8G_PMoEP = mean(EEE8G_PMoEP);
              
for i = 1:data_num
    disp(['For the ',num2str(i),'_th data, ','the ',num2str(I(i)),'_th initialization is selected. ','The number of mixture compoments is ',...
        num2str(KKKK_PMoEP),' and the selected p is ',num2str(PPP{i}),'.'])
end

% disp(["the model.eta is :", num2str(model.eta)," and the model.Pi is :", num2str(model.Pi)]) 

% format short e;
% M_Result_PMoEP = [m_EEE1G_PMoEP m_EEE2G_PMoEP m_EEE3G_PMoEP m_EEE4G_PMoEP...
%                   m_EEE5G_PMoEP m_EEE6G_PMoEP m_EEE7G_PMoEP m_EEE8G_PMoEP]
% disp(['Perforance of the eight measures are ',num2str(M_Result_PMoEP)]);

% O_rating = ceil((U0 * V0' + rating_Ave) .* rating_Ind);
% target_func = (model.eta^(1/0.8) / 2* gamma(1/0.)) * exp^(-model.eta*(abs(x)^0.8));
% fun = inline('(model.eta^(1/0.8) / 2* gamma(1/0.)) * exp^(-model.eta*(abs(x)^0.8))', 'x');
% MAE = sum(sum(abs(X_Noi + rating_Ave - O_rating)))/rating_counts;
toc;
              