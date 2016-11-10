% Initializations
train_examples = load('C:\Users\Laurel\Matlab\CS534\src\cs534_git\Implementation3\iris_train-1.csv');
test_examples = load('C:\Users\Laurel\Matlab\CS534\src\cs534_git\Implementation3\iris_test-1.csv');


 num_examples = size(trainFile,1);
 num_features = size(trainFile,2)-1;   %last column is classification
 classification = num_features + 1;    %column containing classification
 
 informationGain(train_examples, num_features, 3);

%indicies of features: [1,2...4] or other combination of features 
function [weights, costHistory] = informationGain(examples, features, threshold)
    if num_examples < threshold
        return
    end
   
    for i = 1 : size(trainFile,2)
        %put all features into array
    end
    %sorted_features = sort(array)
    
    %find best threshold, use each value in sorted_features as threshold
    %and find best information gain
    
    %return threshold
end

% split data based on threshold calculated in informationGain
% with each split calculate new split threshold with informationGain
