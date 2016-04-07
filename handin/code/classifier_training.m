%% the classifier_training function

function [w, b] = classifier_training(features_pos, features_neg, feature_params)
    pos_len = size(features_pos, 1);
    neg_len = size(features_neg, 1);
    label_pos = ones(pos_len, 1);
    label_neg = -1*ones(neg_len, 1);
    label = [label_pos; label_neg];
    feature = [features_pos', features_neg'];

    [w, b] = vl_svmtrain(feature, label, 1*1e-4);
    save('var_svm_w.mat', 'w');
    save('var_svm_b.mat', 'b');
