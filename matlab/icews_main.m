function [scores] = icews_main(datafile, label)
datafile
data = readtable(datafile, 'Delimiter', '\t');
%data = readtable('/home/weiw/workspace/data/icews/icews_test_data/Iraq_Iraq.csv', 'Delimiter', '\t');
data_arr = table2array(data);
[T, D] = size(data_arr);
L = 10;
scores = [];
for i=L:-1:2
    trainX = data_arr(1:T-i,1:D-1);
    max_t = max(trainX);
    min_t = min(trainX);
    trainX = bsxfun(@rdivide, bsxfun(@minus, trainX, min_t),(max_t - min_t));
    trainY = data_arr(2:T-i+1,D);

    testX = bsxfun(@rdivide, data_arr(T-i+1,1:D-1) - min_t, (max_t - min_t));
    testY = data_arr(T-i+2,D);

    [marg transition emission] = trainIOHMM(trainX, trainY, 3);

    tempX = [testX marg.T'];
    a = tempX * transition.glim{1}.w1 + transition.glim{1}.b1;
    prob = softmax(a');
    prob_idx = argmax(prob);

    pred_y = emission.weights(:,:,prob_idx) * testX' + emission.mean(prob_idx);
    pred_y = ceil(pred_y);
    if pred_y < 0
        pred_y = 0;
    end
    score = evaluation(pred_y, testY);
    r = sprintf('%s --> Preidct %d , real value %d, score: %0.2f ', label, pred_y, testY, score)
    scores = [scores r]
end
end