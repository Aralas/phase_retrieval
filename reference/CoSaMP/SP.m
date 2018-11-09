clc;
clear;

%%
% Y = |Ax|
% Yt = Y.^2
% We want to recover x from Y(or Yt).
%% Settings
ratio = 0.6; % m = ratio * n
n = 100;
method = "Gaussian"; % 还有"0-1"初始化，但是我这里没有用
sparsity = 4;
K = 10; % The sparsity level we esitmate. 我们估计的稀疏度，大于等于sparsity。
isComplex = 0; % 复数1，实数0
tol = 1e-5; % The threhold. 迭代终止阈值
iterNum = 200; % The maximal iteration number. 迭代终止上限次数

%count = 0;
%trialNum = 100;
%for t = 1:trialNum
[X,Y,A] = init(n,ratio, sparsity, isComplex, method);
RealSupport = find(X); %实际支持（调试时候用，接下来的代码中并没有用到）
Yt = abs(Y).^2;
%Xt = X*conj(X).';

%% Tool functions.
Loss = @(A, x, y)norm(LinearOperatorA(A, x*conj(x).')-y,2);
Grad = @(A, x, y)AdjointOperatorA(A, LinearOperatorA(A, x*conj(x).')-y)*x;
Hess = @(A, x, y)AdjointOperatorA(A, LinearOperatorA(A, x*conj(x).'))*2 + ...
    AdjointOperatorA(A, LinearOperatorA(A, x*conj(x).')-y);
% 三个用来简单调用的句柄函数。
%% Here are the codes I used to test DGN.
% 我自己测试DGN的一段代码，看一下多少个初值点能够大概率找到最好的x，一般10个足够。
% count = 0;
% for k = 1:10
%     support = RealSupport;
%
%     Xhat = [];
%     R0 = [];
%     for i = 1:10
%         x0 = randn(n,1);
%         [xhat0,k0] = DGN(Loss, Grad, Hess, A, x0, Yt, support);
%         Xhat = [Xhat xhat0];
%         r0 = norm(abs(A*xhat0).^2-Yt);
%         R0 = [R0 r0];
%     end
%     [~, index] = sort(R0);
%     xhat = Xhat(:,index(1));
%     tempM = X./xhat;
%     tempM = roundn(tempM(~isnan(tempM)),-3);
%     if norm(xhat-X) < 1e-5 || length(unique(tempM)) <= 2
%         count = count+1;
%     else
%         disp(R0)
%         disp(norm(xhat-X))
%     end
% end
% disp(count/10)
%% The algorithm begins.
x0 = 0;%randn(n,1) + 1i*isComplex*randn(n,1);% Initial 0 is also accepted.
grad0 = Grad(A, x0, Yt); %梯度
[~, indexes] = sort(abs(grad0));
support = indexes(end-sparsity+1:end); % 梯度排序
% r0 = Yt - LinearOperatorA(A, x0*conj(x0).'); % The residue.这个残差没有用的..
projM = zeros(n,1); %投影矩阵
projM(support, 1) = 1;
x1 = x0.*projM; % 得到要进入迭代的初值
% 这里第一次迭代因为我们没有初始的support，所以我第一次单独拿出来写了得到support，之后
% 开始循环
% Loops
repeat = 0;
% 记录重复的support，如果上一次support和这一次support相同，则加一，
% 否则归零，达到5次重新随机一个support的元素，避免陷入局部最优
% record = []; %调试用，没别的用处
for k = 1:iterNum
    [~, indexes] = sort(abs(Grad(A, x1, Yt)));
    
    supportcopy = support; % 记录上一次的support
    
    support0 = indexes(end-K+1:end);
    supportX = union(support, support0); % 更新support
    
    % Use DGN to pick up the best support.
    % Randomly set 10 initial points to find the best x.
    Xhat = [];
    R00 = [];
    for i = 1:10
        x2 = randn(n,1) + 1i*isComplex*randn(n,1);
        [xhat0,k0] = DGN(Loss, Grad, Hess, A, x2, Yt, supportX);
        Xhat = [Xhat xhat0];
        r00 = norm(abs(A*xhat0).^2-Yt);
        R00 = [R00 r00];
    end
    [~, index] = sort(R00);
    xhat = Xhat(:,index(1)); % 从十个恢复值中找到最好的xhat
    
    [~, indexes] = sort(abs(xhat));
    support = indexes(end-K+1:end); % 排序得到新的support
    
    
    % 接下来对局部最优进行排查（repeat）
    if isempty(setxor(support,supportcopy))
        repeat = repeat + 1;
        % p = [p 1];
        if repeat >= 5
            try
                support = [indexes(end-K); indexes(end-K+2:end)];
            catch
                s = randperm(n);
                support = randperm(s(1:K));
            end
            repeat = 0;
        end
    else
        % p = [p 0];
        repeat = 0;
    end
    x1 = xhat;
    %     x0 = xhat;
    %     projM = zeros(n,1);
    %     projM(support, 1) = 1;
    %     x1 = x0.*projM; % 更新x1
    %r0 = Yt - LinearOperatorA(A, x1*conj(x1).');
    
    if Loss(A,x1,Yt) < tol
        break
    end
    
end

%A0 = zeros(ratio*n, n);
%A0(:,RealSupport) = A(:,RealSupport)
%Loss(A0,X,Yt)
%end
display(Loss(A,x1,Yt))
%SuccessRatio = count/trialNum;
%display(SuccessRatio)