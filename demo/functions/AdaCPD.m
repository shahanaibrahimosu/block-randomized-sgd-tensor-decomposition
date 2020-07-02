function [ A, MSE_A ,NRE_A, TIME_A] = AdaCPD(X,ops)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AdaCPD Algorithm (Tensor decomposition using randomized block coordiante 
% + stochastic gradient + adaptive stepsize schedule)
% ============== input =====================================
% X  : the data tensor
% ops: algorithm parameters
%   o 'constraint'          - Latent factor constraints
%   o 'b0'                  - adagrad parameter
%   o 'eta'                 - stepsize parameter
%   o 'n_mb'                - Number of fibers
%   o 'max_it'              - Maximum number of iterations
%   o 'A_ini'               - Latent factor initializations
%   o 'A_gt'                - Ground truth latent factors (for MSE computation only)
%   o 'tol'                 - stopping criterion
% =============================================================
% ============= output ========================================
% A: the estimated factors
% MSE_A : the MSE of A at different iterations
% NRE_A : the cost function at different iterations
% TIME_A: the walltime at different iterations
% =============================================================
% Coded by Xiao Fu, Shahana Ibrahim email: (xiao.fu,ibrahish)@oregonstate.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Code
% Get the algorithm parameters
A       = ops.A_ini; 
b0      = ops.b0;
n_mb    = ops.n_mb;
max_it  = ops.max_it;
A_gt    = ops.A_gt;
tol     = ops.tol;
eta     = ops.eta;

% Get the initial parameters
F  = size(A{1},2);
XX = tensor(X);
dim = length(size(X)); % to decide the order of the tensor
dim_vec = size(X);
PP = tensor(ktensor(A)); 
err_e = norm(XX(:) - PP(:),2)^2;
NRE_A(1) =err_e;
MSE_A(1) = (1/3)*(MSE_measure(A{1},A_gt{1})+MSE_measure(A{2},A_gt{2})+MSE_measure(A{3},A_gt{3}));
mmm = 1;
mu=0;
Gt = cell(dim,1);
for nnn=1:dim
    Gt{nnn}=b0;
end
a=tic;
TIME_A(1)=toc(a);

% Run the algorithm until the stopping criterion
for it = 1:max_it 
    % randomly permute the dimensions
    block_vec = randperm(dim);
    % select the block variable to update.  
    d_update = block_vec(1); 

    % sampling fibers and forming the X_{d}=H_{d} A_{d}^t least squares
    [tensor_idx, factor_idx] = sample_fibers(n_mb, dim_vec, d_update);
    % reshape the tensor from the selected samples
    X_sample = reshape(X(tensor_idx), dim_vec(d_update), [])';
    % perform a sampled khatrirao product 
    ii=1;
    for i=[1:d_update-1,d_update+1:dim]
          A_unsel{ii}= A{i};
          ii=ii+1;
    end
    H = sampled_kr(A_unsel, factor_idx);

    % compute the gradient in the current iteration
    g = (1/n_mb)*(A{ d_update }*(H'*H+mu*eye(F))-X_sample'*H-mu*A{d_update});
    % get the accumulated gradient
    Gt{d_update}= ((abs(g).^2)) + Gt{d_update};
    % compute the adaptive stepsize   
    eta_adapted=(eta./sqrt(Gt{d_update}));
    
    % update the selected block
    A{d_update} = A{ d_update} - (eta_adapted).*g;
    A{d_update} = proxr(A{d_update}, ops, d_update);
    
    % compute MSE after each MTTKRP    
    if mod(it,ceil((size(X,1)^2)/n_mb))==0
        TIME_A(mmm+1)= TIME_A(mmm)+toc(a);
        MSE_A(mmm+1)=(1/3)*(MSE_measure(A{1},A_gt{1})+MSE_measure(A{2},A_gt{2})+MSE_measure(A{3},A_gt{3}));%MSE_xiao(A{1},A_gt{1}); % use the first block as performance measure

        P = ktensor(A);
        PP = tensor(P);
        err_e = norm(XX(:) - PP(:))^2;
        NRE_A(mmm+1) = (err_e);
        
        if abs(NRE_A(mmm+1))<=tol
            break;
        end
        
        disp(['AdaCPD at iteration ',num2str(mmm+1),' and the MSE is ',num2str(MSE_A(mmm+1))])
        disp(['AdaCPD at iteration ',num2str(mmm+1),' and the NRE is ',num2str(NRE_A(mmm+1))])
 
        disp('====')
        mmm = mmm + 1;
        a=tic;
    end  
end
end
