function [xhat,k,Dk,Gk] = DGN(Loss, Grad, Hess, A, x, y, support)
% DGN
% y = |Ax|^2
% Loss, Grad, Hess�Ƕ���õĺ������
% ����: Loss, Grad, Hess�������������A����ʼֵx���۲⵽��y���Լ�֧�ż���support
% ���: 
% xhat: ������ֵx��֧�ż�support�µõ���xhat��
% k: DGN��������
% DK: grad*inv(hess)ֵ�����Ե�ʱ�򷽱㿴��ʵ���в����ã��ǲ鿴ֹͣʱ��ֵ��һ��ֹͣʱDK�ǳ��ӽ�0��
% Hk: hessֵ�����Ե�ʱ�򷽱㿴��ʵ���в����ã�

% ����һ��DGN�Ǵ����������ģ�������Ҳ�е�����..
n = length(x);
maxIter = 200; % DGN�������޴�������Ϊ���������Ժã�200���㹻
alpha = 0.4; % DGN����
sigma = 0.8; % DGN����
epsilon = 1e-10; % �ݶ�С��epsilonʱֹͣDGN
A0 = A(:, support); % ��supportͶӰ
x0 = x(support,1); % ��supportͶӰ
% ������DGN����
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

