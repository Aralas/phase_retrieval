function [Y] = AdjointOperatorA(A,y)
% AdjointOperatorA
% АщЫцЫузгA
[m, ~] = size(A);
Y = 0;

for k = 1:m
    a = y(k)*A(k,:).'*A(k,:); 
    Y = Y+a;
end

end

