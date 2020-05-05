function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Theta = 4x3 ; R = 5x4 ; X = 5x3 ; Y = 5x4
% without regularization :: 

% with loop
%for j = 1:num_users
%  J = J + (1/2) * sum((((X * Theta(j,:)') - Y(:,j)).^2) .* R(:,j)); 
%endfor
% without loop
J = (1/2) * sum(sum((((X * Theta') - Y).^2) .* R));

X_grad = (((X * Theta') - Y) .* R) * Theta;

Theta_grad = (((X * Theta') - Y) .* R)' * X;

% adding regularization terminal_size

reg_J = (lambda/2) * ((sum(sum(Theta.^2))) + (sum(sum(X.^2))));
reg_xGrad = lambda * X;
reg_thetaGrad = lambda * Theta;

J = J + reg_J;
X_grad = X_grad + reg_xGrad;
Theta_grad = Theta_grad + reg_thetaGrad;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
