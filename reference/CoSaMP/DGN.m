function [xhat,k,Dk,Gk] = DGN(Loss, Grad, Hess, A, x, y, support)
% DGN
% y = |Ax|^2
% Loss, Grad, Hess是定义好的函数句柄
% 输入: Loss, Grad, Hess函数句柄，矩阵A，初始值x，观测到的y，以及支撑集的support
% 输出: 
% xhat: 给定初值x，支撑集support下得到的xhat，
% k: DGN迭代次数
% DK: grad*inv(hess)值（调试的时候方便看，实际中不调用，是查看停止时的值，一般停止时DK非常接近0）
% Hk: hess值（调试的时候方便看，实际中不调用）

% 下面一段DGN是从网上找来的，我现在也有点忘了..
n = length(x);
maxIter = 200; % DGN迭代上限次数，因为本身收敛性好，200次足够
alpha = 0.4; % DGN参数
sigma = 0.8; % DGN参数
epsilon = 1e-10; % 梯度小于epsilon时停止DGN
A0 = A(:, support); % 作support投影
x0 = x(support,1); % 作support投影
% 下面是DGN迭代
for k = 1:maxIter
    x1 = x0;
    Gk = feval(Grad, A0, x1, y);
    Hk = feval(Hess, A0, x1, y);
    Dk = -Hk\Gk;
    if norm(Dk) < epsilon
        break
    end
    m = 0;
    mk = 0;
    while m<20     
        if feval(Loss, A0, x1+sigma^m*Dk, y)<feval(Loss, A0, x1, y)+ ...
                alpha*sigma^m*conj(Gk).'*Dk
            mk = m;
            break
        end
        m = m+1;
    end
    x0 = x1+sigma^mk*Dk;
end

xhat = zeros(n,1);
xhat(support,1) = x0;
end

