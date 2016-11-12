% Initializations
train_examples = load('D:\Aditya\Desktop\School\OSU\MS\Term 1\CS534 - Machine Learning\Implementation 3\iris_train-1.csv');
test_examples = load('D:\Aditya\Desktop\School\OSU\MS\Term 1\CS534 - Machine Learning\Implementation 3\iris_train-1.csv');


num_examples = size(train_examples,1);
num_features = size(train_examples,2)-1;   %last column is classification
classification_index = num_features + 1;    %column containing classification
features = [1 2 3 4];

returnedTree = decisionTree(train_examples, features, 3);
% Bootstrap aggregation - 2c
L_Values = [5, 10, 15, 20, 25, 30];
for i=L_Values
    for L=1:i
        bootstrapExamples = train_examples(randsample(1:length(train_examples), length(train_examples), true), :);
        %baggedTree(bootstrapExamples, features, 3);
    end
end

function [lessThan, greaterThan] = split(examples, feature, delta)
     lessThan = examples(find(examples(:, feature) <= delta), :);
     greaterThan = examples(find(examples(:, feature) > delta), :);
end

% Using the specified features, recursively and greedily construct a
% decision tree for the provided examples. min_examples is termination
% threshold. {leftTree [feature and delta used] rightTree} is returned.
function tree = decisionTree(examples, features, min_examples)
    % if not enough examples or only one classification, return
    if size(examples, 1) < min_examples || length(unique(examples(:, size(examples, 2)))) == 1
        tree = [];
        return;
    end

    [maxGain, maxFeature, maxDelta] = informationGain(examples, features);
    [lessThan, greaterThan] = split(examples, maxFeature, maxDelta);
    tree = {decisionTree(lessThan, features, min_examples), 
            [maxFeature, maxDelta],
            decisionTree(greaterThan, features, min_examples)};
end

% Using the specified features, recursively and greedily construct a
% decision tree for the provided examples using two random features at each step. 
% min_examples is termination threshold. [leftTree [feature and delta used] rightTree] is returned.
function tree = baggedTree(examples, features, min_examples)
    % if not enough examples or only one classification, return
    if size(examples, 1) < min_examples || length(unique(examples(:, size(examples, 2)))) == 1
        tree = [];
        return;
    end
    
    feature_perm = features(randperm(length(features), 2));
    [maxGain, maxFeature, maxDelta] = informationGain(examples, feature_perm);
    [lessThan, greaterThan] = split(examples, maxFeature, maxDelta);
    tree = {baggedTree(lessThan, features, min_examples), 
            [maxFeature, maxDelta],
            baggedTree(greaterThan, features, min_examples)};
end
 
%indicies of features: [1,2...4] or other combination of features 
% Examples comprise what is being considered at the current split
function [maxGain, maxFeature, maxDelta] = informationGain(examples, features)
    classification_index = size(examples, 2);
    num_examples = size(examples, 1);
    
    maxGain = 0;
    maxFeature = 0;
    maxDelta = 0;
    
    classifiers = unique(examples(:, classification_index));
    num_classifiers = length(classifiers);
    originalUncertainty = 0;
   
    tempMatrix = zeros(2, num_classifiers);
    for feature = features
        %todo: generalize for multiple classifications
        deltas = unique(examples(:, feature));
        for i=1:size(deltas,1)
            %set temporary matrices 
            count_array = zeros(3,num_classifiers);
            count_array(1, :) = classifiers';
            occurenceMatrix = zeros(2,num_classifiers);
            
            % returns indices - fix - make a split function
            [lessThan, greaterThan] = split(examples, feature, deltas(i));

            uncertaintyLesser = 0;
            uncertaintyGreater = 0;
            pLess=0;
            pGreater=0;
            
            %count number of class A and class B
            for j = 1 : size(lessThan,1)
                ni = find(count_array(1, :) == lessThan(j, classification_index), classification_index);   %column index of count array containing that classifier
                count_array(2, ni) = count_array(2, ni) + 1;
            end
            for j = 1 : size(greaterThan,1)
                ni = find(count_array(1, :) == greaterThan(j, classification_index), classification_index);   %column index of count array containing that classifier
                count_array(3, ni) = count_array(3, ni) + 1;
            end
            %after this code, we have matrix of class counts of greater
            %than & less than from our initial split
            %We can use these values to calculate information gain
            
            %Counts for classifiers before split
            for j = 1 : num_classifiers
                occurenceMatrix(2, j) = count_array(2, j) + count_array(3, j);
            end
            
            %calculate uncertainty before split
            if originalUncertainty == 0
                for j=1:num_classifiers
                    prob = occurenceMatrix(2, j)/sum(occurenceMatrix(2,:));
                    uncertainty = prob * (log2(prob));
                    originalUncertainty = originalUncertainty - uncertainty;
                end
            end
            
            
            %results in a matrix of uncertainties for 
            %each split
            for j=1:num_classifiers
                numLessThan = sum(count_array(2,:));
                numGreaterThan = sum(count_array(3,:));
                tempLess = count_array(2, j)/numLessThan;
                tempGreater = count_array(3, j)/numGreaterThan;
                tempMatrix(2, j) = (0-tempLess) * log2(tempLess);
                tempMatrix(3, j) = (0-tempGreater) * log2(tempGreater);
            end
            tempMatrix(isnan(tempMatrix)) = 0;
            uncertaintyLesser = sum(tempMatrix(2, :));
            uncertaintyGreater = sum(tempMatrix(3, :));
            pLess = sum(count_array(2,:))/num_examples;
            pGreater=sum(count_array(3,:))/num_examples;
            
            gain = originalUncertainty - (pLess*uncertaintyLesser + pGreater*uncertaintyGreater);

            
            if (gain > maxGain)
                maxFeature = feature;
                maxGain = gain;
                maxDelta = deltas(i);
            end
        end
    end 
end



% split data based on threshold calculated in informationGain
% with each split calculate new split threshold with informationGain
