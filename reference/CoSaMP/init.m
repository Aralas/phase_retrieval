function [X,Y,A] = init(n,ratio, sparsity, isComplex, method)
% init
% �����ʼ������
% ���������SP.m���Ѿ��������
% �����X,A,Y���� Y = AX


m = n*ratio;
A = randn(m, n)/(sqrt(n)); % �ֲ�����N(0,1/n)

switch method
    case "Gaussian"
        x = randn(n,1)+1i*isComplex*randn(n,1);
    case "0-1"
        x = ones(n,1);
end

loc = randperm(n);
x(loc(sparsity+1:end)) = 0;
X = x;
Y = A*x;
end

