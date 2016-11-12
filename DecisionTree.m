% Initializations
train_examples = load('D:\Aditya\Desktop\School\OSU\MS\Term 1\CS534 - Machine Learning\Implementation 3\iris_train-1.csv');
test_examples = load('D:\Aditya\Desktop\School\OSU\MS\Term 1\CS534 - Machine Learning\Implementation 3\iris_test-1.csv');


num_examples = size(train_examples,1);
num_features = size(train_examples,2)-1;   %last column is classification
classification_index = num_features + 1;    %column containing classification
features = [1 2 3 4];

x = 50;
returnedTree = decisionTree(train_examples, features, x, 1);
testTree(returnedTree, test_examples);
% Bootstrap aggregation - 2c
L_Values = [5, 10, 15, 20, 25, 30];
averageTrain = zeros(6, 1);
averageTest = zeros(6, 1);
counter = 0;

for i=L_Values
    counter = counter + 1;
    for j=1:10
        trees = [];
        for L=1:i
            bootstrapExamples = train_examples(randsample(1:length(train_examples), length(train_examples), true), :);
            trees = [trees; {baggedTree(bootstrapExamples, features, x)}];
        end
        averageTrain(counter) = averageTrain(counter) + aggregateTest(trees, train_examples, classification_index);
        averageTest(counter) = averageTest(counter) + aggregateTest(trees, test_examples, classification_index);
    end
    averageTrain(counter) = averageTrain(counter) / 10;
    averageTest(counter) = averageTest(counter) / 10;
end

function [lessThan, greaterThan] = split(examples, feature, delta)
     lessThan = examples(find(examples(:, feature) <= delta), :);
     greaterThan = examples(find(examples(:, feature) > delta), :);
end

% Given a data point and a decision tree, predict the classification
function prediction = predict(tree, example)
    % If at leaf node, return classification
    if length(tree{2}) == 1
        prediction = tree{2};
    % else if the given feature is less than the test and threshold given,
    % go left
    elseif example(tree{2}(1)) <= tree{2}(2)
        prediction = predict(tree{1}, example);
    else %go right
        prediction = predict(tree{3}, example);
    end     
end

% Test the accuracy of a decision tree on a set of examples.
function precision = testTree(tree, data)
    correct = 0;
    for i=1:size(data, 1)
        example = data(i,:);
        prediction = predict(tree, example);
        if prediction == example(size(example, 2))
            correct = correct + 1;
        end
    end
    precision = correct / size(data, 1);
end

% Using a set of concatenated trees, vote to predict the example.
function precision = aggregateTest(trees, data, num_classifications)
    correct = 0;
    for example_index=1:size(data, 1)
        example = data(example_index, :);
        tally = zeros(num_classifications, 1);
        for i=1:size(trees,1)
            tree = trees{i};
            temp_prediction = predict(tree, example);
            tally(temp_prediction) = tally(temp_prediction) + 1;
        end
        [~, prediction] = find(tally == max(tally)); % index is the winning class label
        if prediction == example(num_classifications)
            correct = correct + 1;
        end
    end
    precision = correct / size(data, 1);
end

% Using the specified features, recursively and greedily construct a
% decision tree for the provided examples. min_examples is termination
% threshold. {leftTree [feature and delta used] rightTree} is returned.
function tree = decisionTree(examples, features, min_examples, flag)
    % if not enough examples or only one classification, return
    if size(examples, 1) < min_examples || length(unique(examples(:, size(examples, 2)))) == 1
        tabulated = tabulate(examples(:,5));
        [~, label] = max(tabulated(:,2));
        tree = {[]; label; []};
        return;
    end

    [maxGain, maxFeature, maxDelta] = informationGain(examples, features, flag);
    [lessThan, greaterThan] = split(examples, maxFeature, maxDelta);
    tree = {decisionTree(lessThan, features, min_examples,0); 
            [maxFeature, maxDelta];
            decisionTree(greaterThan, features, min_examples,0)};
end

% Using the specified features, recursively and greedily construct a
% decision tree for the provided examples using two random features at each step. 
% min_examples is termination threshold. [leftTree [feature and delta used] rightTree] is returned.
function tree = baggedTree(examples, features, min_examples, flag)
    % if not enough examples or only one classification, return
    if size(examples, 1) < min_examples || length(unique(examples(:, size(examples, 2)))) == 1
        tabulated = tabulate(examples(:,5));
        [~, label] = max(tabulated(:,2));
        tree = {[]; label; []};
        return;
    end
    
    feature_perm = features(randperm(length(features), 2));
    [maxGain, maxFeature, maxDelta] = informationGain(examples, feature_perm, 0);
    if(maxFeature == 0 || maxDelta == 0)
        tabulated = tabulate(examples(:,5));
        [~, label] = max(tabulated(:,2));
        tree = {[]; label; []};
        return;
    end
    [lessThan, greaterThan] = split(examples, maxFeature, maxDelta);
    tree = {baggedTree(lessThan, features, min_examples, 0); 
            [maxFeature, maxDelta];
            baggedTree(greaterThan, features, min_examples,0)};
end
 
%indicies of features: [1,2...4] or other combination of features 
% Examples comprise what is being considered at the current split
function [maxGain, maxFeature, maxDelta] = informationGain(examples, features, flag)
   
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
        if flag == 1
            featureTable = [feature,feature];
            dataTable = [feature,feature];
        end
        %todo: generalize for multiple classifications
        %deltas = unique(examples(:, feature));
        
        deltas = [examples(:, feature),examples(:, classification_index)];
        deltas = sortrows(deltas, 1);
        
        for i=2:size(deltas,1)    


        threshold = deltas(i,1);

        %set temporary matrices  
        count_array = zeros(3,num_classifiers);
        count_array(1, :) = classifiers';
        occurenceMatrix = zeros(2,num_classifiers);

        % returns indices - fix - make a split function
        [lessThan, greaterThan] = split(examples, feature, threshold);

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
        
       % dummy = [deltas(i), gain];
       % dataTable = [dataTable;dummy];
       
        if (gain > maxGain) %& (deltas(i, 2) ~= deltas(i-1, 2)) 
            maxFeature = feature;
            maxGain = gain;
            maxDelta = deltas(i);    
           % if flag == 1 & deltas(i, 2) ~= deltas(i-1, 2)
            %    dummy = [maxDelta, maxGain];
             %   featureTable=[featureTable;dummy];
            %end 
        end  
        
        if flag == 1 & deltas(i, 2) ~= deltas(i-1, 2)
            dummy = [deltas(i), gain];
            featureTable=[featureTable;dummy];
        end
        
        end
    if flag == 1
        featureTable
       % dataTable
    end
    end
end