% Initializations
train_examples = load('C:\Users\Laurel\Matlab\CS534\src\cs534_git\Implementation3\iris_train-1.csv');
test_examples = load('C:\Users\Laurel\Matlab\CS534\src\cs534_git\Implementation3\iris_test-1.csv');


num_examples = size(train_examples,1);
num_features = size(train_examples,2)-1;   %last column is classification
classification_index = num_features + 1;    %column containing classification
%counting the number of classes in the dataset
%classifiers = unique(train_examples(:, classification_index))
%numClassifiers = length(classifiers);

informationGain(train_examples, [1 2 3 4], 3);
 
%indicies of features: [1,2...4] or other combination of features 
% Examples comprise what is being considered at the current split
function [weights, costHistory] = informationGain(examples, features, threshold)
    if size(examples, 1) < threshold
        return
    end
    
    classification_index = size(examples, 2);
    num_examples = size(examples, 1);
    
    maxGain = 0;
    maxGainFeature = 0;
    maxDelta = 0;
    
    classifiers = unique(examples(:, classification_index));
    num_classifiers = length(classifiers);
    count_array = zeros(3,num_classifiers);
    count_array(1, :) = classifiers';
    for feature = features
        %todo: generalize for multiple classifications
        i=1;
        for delta = unique(examples(:, feature))
            % returns indices - fix
            lessThan = find(examples(:, feature) <= delta(i));
            greaterThan = find(examples(:, feature) > delta(i));
            
            %count number of class A and class B
            for j = i : length(lessThan)
                ni = find(count_array(1, :) == examples(lessThan(j),classification_index))   %column index of count array containing that classifier
                count_array(2, ni) = count_array(2, ni) + 1
            end
            for j = i : length(greaterThan)
                ni = find(count_array(1, :) == examples(greaterThan(j),classification_index))   %column index of count array containing that classifier
                count_array(3, ni) = count_array(3, ni) + 1
            end
            %after this code, we have matrix of class counts of greater
            %than & less than from our initial split
            %We can use these values to calculate information gain
            
            if (gain > maxGain)
                maxGainFeature = feature;
                maxGain = gain;
                maxDelta = delta;
            end
            i = i + 1;
        end
        
    end 
    weights = 0;
    costHistory = 0;
    
    return 
end

% split data based on threshold calculated in informationGain
% with each split calculate new split threshold with informationGain
