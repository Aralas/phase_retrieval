function y = LinearOperatorA(A, X)
% LinearOperatorA
% ��������A
[m, ~] = size(A);
y = zeros(m,1);

for k = 1:m
    y(k) = A(k,:)*X*A(k,:).'; 
end

end

