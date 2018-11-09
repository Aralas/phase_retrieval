clc;
clear;

%%
% Y = |Ax|
% Yt = Y.^2
% We want to recover x from Y(or Yt).
%% Settings
ratio = 0.6; % m = ratio * n
n = 100;
method = "Gaussian"; % ����"0-1"��ʼ��������������û����
sparsity = 4;
K = 10; % The sparsity level we esitmate. ���ǹ��Ƶ�ϡ��ȣ����ڵ���sparsity��
isComplex = 0; % ����1��ʵ��0
tol = 1e-5; % The threhold. ������ֹ��ֵ
iterNum = 200; % The maximal iteration number. ������ֹ���޴���

%count = 0;
%trialNum = 100;
%for t = 1:trialNum
[X,Y,A] = init(n,ratio, sparsity, isComplex, method);
RealSupport = find(X); %ʵ��֧�֣�����ʱ���ã��������Ĵ����в�û���õ���
Yt = abs(Y).^2;
%Xt = X*conj(X).';

%% Tool functions.
Loss = @(A, x, y)norm(LinearOperatorA(A, x*conj(x).')-y,2);
Grad = @(A, x, y)AdjointOperatorA(A, LinearOperatorA(A, x*conj(x).')-y)*x;
Hess = @(A, x, y)AdjointOperatorA(A, LinearOperatorA(A, x*conj(x).'))*2 + ...
    AdjointOperatorA(A, LinearOperatorA(A, x*conj(x).')-y);
% ���������򵥵��õľ��������
%% Here are the codes I used to test DGN.
% ���Լ�����DGN��һ�δ��룬��һ�¶��ٸ���ֵ���ܹ�������ҵ���õ�x��һ��10���㹻��
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
grad0 = Grad(A, x0, Yt); %�ݶ�
[~, indexes] = sort(abs(grad0));
support = indexes(end-sparsity+1:end); % �ݶ�����
% r0 = Yt - LinearOperatorA(A, x0*conj(x0).'); % The residue.����в�û���õ�..
projM = zeros(n,1); %ͶӰ����
projM(support, 1) = 1;
x1 = x0.*projM; % �õ�Ҫ��������ĳ�ֵ
% �����һ�ε�����Ϊ����û�г�ʼ��support�������ҵ�һ�ε����ó���д�˵õ�support��֮��
% ��ʼѭ��
% Loops
repeat = 0;
% ��¼�ظ���support�������һ��support����һ��support��ͬ�����һ��
% ������㣬�ﵽ5���������һ��support��Ԫ�أ���������ֲ�����
% record = []; %�����ã�û����ô�
for k = 1:iterNum
    [~, indexes] = sort(abs(Grad(A, x1, Yt)));
    
    supportcopy = support; % ��¼��һ�ε�support
    
    support0 = indexes(end-K+1:end);
    supportX = union(support, support0); % ����support
    
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
    xhat = Xhat(:,index(1)); % ��ʮ���ָ�ֵ���ҵ���õ�xhat
    
    [~, indexes] = sort(abs(xhat));
    support = indexes(end-K+1:end); % ����õ��µ�support
    
    
    % �������Ծֲ����Ž����Ų飨repeat��
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
    %     x1 = x0.*projM; % ����x1
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