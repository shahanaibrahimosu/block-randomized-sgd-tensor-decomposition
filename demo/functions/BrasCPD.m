function [ A, MSE_A ,NRE_A, TIME_A] = BrasCPD(X,ops)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BrasCPD Algorithm (Tensor decomposition using randomized block coordiante 
% + stochastic gradient)
% ============== input =====================================
% X  : the data tensor
% ops: algorithm parameters
%   o 'constraint'          - Latent factor constraints
%   o 'b0'                  - Initial stepsize
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
% Coded by Xiao Fu, Shahana Ibrahim, email: (xiao.fu,ibrahish)@oregonstate.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Code
% Get the algorithm parameters
A       = ops.A_ini; 
b0      = ops.b0;
n_mb    = ops.n_mb;
max_it  = ops.max_it;
A_gt    = ops.A_gt;
tol     = ops.tol;

% Get initial parametrs
dim = length(size(X)); 
dim_vec = size(X);
PP = tensor(ktensor(A)); 
XX = tensor(X);
err_e = norm(XX(:) - PP(:),2)^2;
NRE_A(1) = (err_e);
MSE_A(1) = (1/3)*(MSE_measure(A{1},A_gt{1})+MSE_measure(A{2},A_gt{2})+MSE_measure(A{3},A_gt{3}));
mmm = 1;
a=tic;
TIME_A(1)=toc(a);

% Run the algorithm until the stopping criterion
for it = 1:max_it 
    % step size 
    alpha =b0/(n_mb*(it)^(1e-6));   
    %randomly permute the dimensions
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
   
    % update the selected block
    alpha_t = alpha;      
    d = d_update;   
    A{d_update} = A{d_update} - alpha_t*(A{ d_update }*H'*H-X_sample'*H);
    A{d_update} = proxr(A{d_update}, ops, d);
     
    % compute MSE after each MTTKRP
    if mod(it,ceil((size(X,1)^2)/n_mb))==0
        TIME_A(mmm+1)= TIME_A(mmm)+toc(a);
        MSE_A(mmm+1)=(1/3)*(MSE_measure(A{1},A_gt{1})+MSE_measure(A{2},A_gt{2})+MSE_measure(A{3},A_gt{3}));
        P = ktensor(A);
        PP = tensor(P);
        NRE_A(mmm+1) = norm(XX(:) - PP(:))^2;
    
        if abs(NRE_A(mmm+1))<=ops.tol
            break;
        end
        
        disp(['BrasCPD at iteration ',num2str(mmm+1),' and the MSE is ',num2str(MSE_A(mmm+1))])
        disp(['at iteration ',num2str(mmm+1),' and the NRE is ',num2str(NRE_A(mmm+1))])
        disp('====')
        mmm = mmm + 1;
        a=tic;
    end  
end   
end

