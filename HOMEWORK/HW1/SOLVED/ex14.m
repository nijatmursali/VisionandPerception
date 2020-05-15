clear; clc

w = 1;

x1 = [4.39267426e+02 2.12330383e+02 w].';
x2 = [6.45424927e+02 4.10001312e+02 w].';
x3 = [2.19429459e+02 1.23201721e+03 w].';
x4 = [6.47419067e+02 6.06201294e+02 w].';
x5 = [2.28142853e+02 8.02071411e+02 w].';
x6 = [2.42005367e+01 4.03327515e+02 w].';
x7 = [4.39924225e+02 1.22728711e+03 w].';
x8 = [4.39592438e+02 4.07886230e+02 w].';
x = [x1 x2 x3 x4 x5 x6 x7 x8];

X1 = [6 3 0 w].';
X2 = [9 6 0 w].';
X3 = [3 18 0 w].';
X4 = [9 9 0 w].';
X5 = [3 12 0 w].';
X6 = [0 6 0 w].';
X7 = [6 18 0 w].';
X8 = [6 6 0 w].';
X = [X1 X2 X3 X4 X5 X6 X7 X8];

n_correspondences = size(X, 2);

for i=1:n_correspondences
    X_tmp = X(:, i);
    X(3, i) = sqrt(20^2 + X_tmp(1)^2 + X_tmp(2)^2);
end

C_X = sum(X.')/n_correspondences;
C_x = sum(x.')/n_correspondences;

X_variance = 0;
Y_variance = 0;
Z_variance = 0;

x_variance = 0;
y_variance = 0;

for i=1:n_correspondences
    X_tmp = X(:, i);
    X_variance = X_variance + (C_X(1) - X_tmp(1))^2;
    Y_variance = Y_variance + (C_X(2) - X_tmp(2))^2;
    Z_variance = Z_variance + (C_X(3) - X_tmp(3))^2;
    
    x_tmp = x(:, i);
    x_variance = x_variance + (C_x(1) - x_tmp(1))^2;
    y_variance = y_variance + (C_x(2) - x_tmp(2))^2;
end

X_variance = X_variance / n_correspondences;
Y_variance = Y_variance / n_correspondences;
Z_variance = Z_variance / n_correspondences;

X_scaling = sqrt(3)/X_variance;
Y_scaling = sqrt(3)/Y_variance;
Z_scaling = sqrt(3)/Z_variance;

T_X = [X_scaling 0 0 -X_scaling*C_X(1);
       0 Y_scaling 0 -Y_scaling*C_X(2);
       0 0 Z_scaling -Z_scaling*C_X(3);
       0 0 0 1];

x_variance = x_variance / n_correspondences;
y_variance = y_variance / n_correspondences;

x_scaling = sqrt(2)/x_variance;
y_scaling = sqrt(2)/y_variance;

T_x = [x_scaling 0 -x_scaling*C_x(1);
       0 y_scaling -y_scaling*C_x(2);
       0 0 1];

X_norm = T_X*X;
x_norm = T_x*x;

A = zeros(n_correspondences*2, 12);
index = 1;

for i=1:n_correspondences
    X_tmp = X_norm(:, i);
    x_tmp = x_norm(:, i);
    
    A(index, :) = [zeros(1, 4) -x_tmp(3)*X_tmp.' x_tmp(2)*X_tmp.'];
    A(index + 1, :) = [x_tmp(3)*X_tmp.' zeros(1, 4) -x_tmp(1)*X_tmp.'];
    index = index + 2;
end

H = A.'*A;

[V, D] = eig(H);

p = V(:, 1);
P_norm = [p(1:4).'; p(5:8).'; p(9:12).'];

P = T_x\P_norm*T_X;

err_tot = 0;

for i=1:n_correspondences
    x_itmp = P * X(:, i);
    x_itmp = x_itmp/x_itmp(3);
    
    err = sqrt((x(:, i) - x_itmp).'*(x(:, i) - x_itmp))
    err_tot = err_tot + err;
end

M = P(1:3, 1:3);

% Compute K, R and t using QR-Decomposition.
[R, K] = qr(M);

t = R*M\P(:, 4);