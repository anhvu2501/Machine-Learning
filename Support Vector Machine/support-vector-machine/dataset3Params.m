function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

results = zeros(64,3); % result matrix to store 8^2 different results of 8^2 different models
error_row = 0; % mark the current error row when running loops (just like an index)

C_test = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_test = [0.01 0.03 0.1 0.3 1 3 10 30];
for i = 1 : size(C_test, 2),
   for j = 1 : size(sigma_test, 2),
     error_row = error_row + 1; % increase by 1 for each loop
     model = svmTrain(X, y, C_test(i), @(x1, x2) gaussianKernel(x1, x2, sigma_test(j))); % train SVM model by using Gaussian Kernel
     predictions = svmPredict(model, Xval); % The vector contains all the predictions from the SVM (predict the label on the cross validation set)
     prediction_error = mean(double(predictions ~= yval)); % for evaluating the error on the cross validation set 
     % For classification, the error is defined as the fraction of the cross validation examples that were classified incorrectly
     results(error_row, :) = [C_test(i), sigma_test(j), prediction_error]; % store each C_test, sigma_test and the prediction_error correspondingly
   end
end


sorted_results = sortrows(results, 3); % sort matrix by column 3 ascendingly (sort by prediction_error)

% Then we can have the values for C and sigma
C = sorted_results(1, 1);
sigma = sorted_results(1, 2);

% =========================================================================

end
