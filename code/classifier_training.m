%% the classifier_training function

function [w, b] = classifier_training(features_pos, features_neg, feature_params)
    if exist('var_svm_m.mat', 'file') && exist('var_svm_b.mat', 'file')
        load('var_svm_m.mat');
        load('var_svm_b.mat');
    else
        [pos_len, ~] = size(features_pos);
        [neg_len, ~] = size(features_neg);
        label = ones(pos_len + neg_len, 1);
        label(neg_len:end) = -1*label(neg_len:end);
        feature = [features_pos', features_neg'];
        [w, b] = vl_svmtrain(feature, label, 0.0001);
        save('var_svm_m.mat', 'w');
        save('var_svm_b.mat', 'b');
    end

