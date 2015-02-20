% Sigmoid Belief IOHMM
% Here is the model
%
%  X \  X \
%  | |  | |
%  Q-|->Q-|-> ...
%  | /  | /
%  Y    Y
%
function [marg, transition, emission] = trainIOHMM(trainX, trainY, n)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
X = 1; Q = 2; Y =3;
%intra time slice graph
intra = zeros(3);
intra(X, [Q Y]) = 1;
intra(Q, Y) = 1;

% inter time slice graph
inter = zeros(3);
inter(Q, Q) = 1;

%define the number of 
ns = [size(trainX, 2), n, 1];
dnodes = [Q]; %define the list of discrete node
eclass1 = [1 2 3];
eclass2 = [1 4 3];
bnet = mk_dbn(intra, inter, ns, dnodes, eclass1, eclass2);
bnet.CPD{1} = root_CPD(bnet, 1);
% ==========================================================
bnet.CPD{2} = softmax_CPD(bnet, 2);
bnet.CPD{4} = softmax_CPD(bnet, 5, 'discrete', [2]);
% ==========================================================
bnet.CPD{3} = gaussian_CPD(bnet, 3);

% construct the training dataset
T = size(trainX, 1);
cases = cell(3, T);
cases(1,:) = num2cell(trainX', 1);
cases(3,:) = num2cell(trainY);
engine = bk_inf_engine(bnet);

% log lik before learning
[engine, loglik] = enter_evidence(engine, cases);

% do learning
ev=cell(1,1);
ev{1}=cases;
[bnet2, LL2] = learn_params_dbn_em(engine, ev);

% infer the most likely last status of X
trainedEngine = bk_inf_engine(bnet2);
[trainedEngine, ll] = enter_evidence(trainedEngine, cases);
marg = marginal_nodes(trainedEngine, 2, T);
transition = struct(bnet2.CPD{4});
emission = struct(bnet2.CPD{3});
end

