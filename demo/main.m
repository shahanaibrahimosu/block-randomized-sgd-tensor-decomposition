clear;
clc;
close all

% add paths
addpath functions
addpath tensor_toolbox
addpath tensorlab_2016-03-28

% Problem setup
X = {};                     % Input tensor
F = 10;                     % Rank
iter_mttkrp = 30;           % Number of MTTKRPs
I_vec = [100];              % Tensor size
bs = [18];      % Number of fibers
num_trial=2;                % Number of trials

for i1 = 1:length(I_vec)

for trial = 1:num_trial
    
    
    disp('======================================================================================')
    disp(['running at trial ',num2str(trial), ': I equals ' ,num2str(I_vec(i1)), ' and F equals ' ,num2str(F)])
    disp('======================================================================================')
    
    I{1} = I_vec(i1);
    I{2} = I_vec(i1);
    I{3} = I_vec(i1);
    
    % Generate the true latent factors
    for i=1:3
        A{i} = (rand(I{i},F));
    end
    A_gt = A;
    
    % Form the tensor
    for k=1:I{3}
        X{i1}(:,:,k)=A{1}*diag(A{3}(k,:))*A{2}';
    end
    XX = tensor(X{i1});
    X_data = XX;
    
    % Initialize the latent factors   
    for d = 1:3
        Hinit{d} = rand( I{d}, F );
    end
     

    %% BrasCPD stepsize 1  
    ops.constraint{1} = 'nonnegative';
    ops.constraint{2} = 'nonnegative';
    ops.constraint{3} = 'nonnegative';
    ops.b0 = 0.1;
    ops.n_mb = bs(i1);
    ops.max_it = (I{1}*I{2}/ops.n_mb)*iter_mttkrp;
    ops.A_ini = Hinit;
    ops.A_gt=A_gt; % use the ground truth value for MSE computation
    ops.tol= eps^2;
    [ A_1, MSE_A_1 ,NRE_A_1,TIME_A_1] = BrasCPD(X_data,ops);
    len = length(MSE_A_1);
    MSE_Xiao1{i1}(trial,:)= [MSE_A_1 zeros(1,len-iter_mttkrp)];
    NRE_Xiao1{i1}(trial,:)= [NRE_A_1 zeros(1,len-iter_mttkrp)];
    TIME_Xiao1{i1}(trial,:)=[TIME_A_1 zeros(1,len-iter_mttkrp)];
    
    
   
    %% BrasCPD stepsize 2
    ops.constraint{1} = 'nonnegative';
    ops.constraint{2} = 'nonnegative';
    ops.constraint{3} = 'nonnegative';
    ops.b0 = 0.05;
    ops.n_mb = bs(i1);
    ops.max_it = (I{1}*I{2}/ops.n_mb)*iter_mttkrp;
    ops.A_ini = Hinit;
    ops.A_gt=A_gt; % use the ground truth value for MSE computation
    ops.tol= eps^2;
    [ A_2, MSE_A_2 ,NRE_A_2,TIME_A_2] = BrasCPD(X_data,ops);
    len = length(MSE_A_2);
    MSE_Xiao2{i1}(trial,:)= [MSE_A_2 zeros(1,len-iter_mttkrp)];
    NRE_Xiao2{i1}(trial,:)= [NRE_A_2 zeros(1,len-iter_mttkrp)];
    TIME_Xiao2{i1}(trial,:)=[TIME_A_2 zeros(1,len-iter_mttkrp)];

    
    
    %% BrasCPD stepsize 3
    ops.constraint{1} = 'nonnegative';
    ops.constraint{2} = 'nonnegative';
    ops.constraint{3} = 'nonnegative';
    ops.b0 = 0.01;
    ops.n_mb = bs(i1);
    ops.max_it = (I{1}*I{2}/ops.n_mb)*iter_mttkrp;
    ops.A_ini = Hinit;
    ops.A_gt=A_gt; % use the ground truth value for MSE computation
    ops.tol= eps^2;
    [ A_3, MSE_A_3 ,NRE_A_3,TIME_A_3] = BrasCPD(X_data,ops);
    len = length(MSE_A_3);
    MSE_Xiao3{i1}(trial,:)= [MSE_A_3 zeros(1,len-iter_mttkrp)];
    NRE_Xiao3{i1}(trial,:)= [NRE_A_3 zeros(1,len-iter_mttkrp)];
    TIME_Xiao3{i1}(trial,:)=[TIME_A_3 zeros(1,len-iter_mttkrp)];
    
    
    
    %% AdaCPD
    ops.constraint{1} = 'nonnegative';
    ops.constraint{2} = 'nonnegative';
    ops.constraint{3} = 'nonnegative';
    ops.eta = 1;
    ops.b0 = 1;
    ops.n_mb = bs(i1);
    ops.max_it = (I{1}*I{2}/ops.n_mb)*iter_mttkrp;
    ops.A_ini = Hinit;
    ops.A_gt=A_gt; % use the ground truth value for MSE computation
    ops.tol= eps^2;
    [ A_ada, MSE_A_adagrad ,NRE_A_adagrad, TIME_A_adagrad] = AdaCPD(X_data,ops);
    len = length(MSE_A_adagrad);   
    MSE_Xiao_adagrad{i1}(trial,:) = [MSE_A_adagrad zeros(1,len-iter_mttkrp)];
    NRE_Xiao_adagrad{i1}(trial,:)= [NRE_A_adagrad zeros(1,len-iter_mttkrp)];
    TIME_Xiao_adagrad{i1}(trial,:)= [TIME_A_adagrad zeros(1,len-iter_mttkrp)];
 
    
end

    %% plot
    figure(i1+10)
    semilogy([0:(size(MSE_Xiao1{i1},2)-1)],mean(MSE_Xiao1{i1},1),'-sb','linewidth',1.5);hold on
    semilogy([0:(size(MSE_Xiao2{i1},2)-1)],mean(MSE_Xiao2{i1},1),'-ob','linewidth',1.5);hold on
    semilogy([0:(size(MSE_Xiao3{i1},2)-1)],mean(MSE_Xiao3{i1},1),'->b','linewidth',1.5);hold on
    semilogy([0:(size(MSE_Xiao_adagrad{i1},2)-1)],mean(MSE_Xiao_adagrad{i1},1),'-dg','linewidth',1.5);hold on
    legend('BrasCPD (\alpha = 0.1)','BrasCPD (\alpha = 0.05)','BrasCPD (\alpha = 0.01)','AdaCPD')
    xlabel('no. of MTTKRP computed')
    ylabel('MSE')
    set(gca,'fontsize',14)
    grid on
  
    figure(i1+100)
    semilogy([0:(size(NRE_Xiao1{i1},2)-1)],mean(NRE_Xiao1{i1},1)/prod(size(XX)),'-sb','linewidth',1.5);hold on
    semilogy([0:(size(NRE_Xiao2{i1},2)-1)],mean(NRE_Xiao2{i1},1)/prod(size(XX)),'-ob','linewidth',1.5);hold on
    semilogy([0:(size(NRE_Xiao3{i1},2)-1)],mean(NRE_Xiao3{i1},1)/prod(size(XX)),'->b','linewidth',1.5);hold on
    semilogy([0:(size(NRE_Xiao_adagrad{i1},2)-1)],mean(NRE_Xiao_adagrad{i1},1)/prod(size(XX)),'-dg','linewidth',1.5);hold on
    legend('BrasCPD (\alpha = 0.1)','BrasCPD (\alpha = 0.05)','BrasCPD (\alpha = 0.01)','AdaCPD')
    xlabel('no. of MTTKRP computed')
    ylabel('Cost')
    set(gca,'fontsize',14)
    grid on

end    
     
     




