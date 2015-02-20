function [score] = evaluation(pred, actual)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
occu_score = 0.5 * ((pred > 0) == (actual > 0));
dem = max([pred actual 4]);
dim = abs(pred - actual);
accr_score = 3.5 * (1 - dim/dem);
score = round((accr_score + occu_score) * 100) / 100;
end

