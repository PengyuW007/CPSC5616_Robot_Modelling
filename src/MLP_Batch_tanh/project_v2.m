%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% COURSE: CPSC 5616
% MEMBER: 
%       Pengyu Wang, 0425157
%       Haokun Zhang, 0424660
%       Ziping Zhu, 0422426
% PROJECT: Robot Modelling by Using MLP Batch
%
% REMARKS: Prediction of robot motion
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
close all;
clear;
tic

file = 3;

if file == 1
    data = readtable('Robot Dataset_with_6 inputs and 2 Outputs.xlsx');
elseif file == 2
    data = readtable('Dataset_5000.xlsx');
elseif file == 3
    data = readtable('Dataset_300000.xlsx');
end

data = data{:,:};

%x = 1:0.1:100;
%w = 1:0.1:100;
%data = [x',w',sin(x+w)',(x+w)', 2*(x+w)'];

% Split the data into training, validation, and testing sets
numData = size(data, 1);
numTrain = round(0.6*numData);
numVal = round(0.2*numData);
numTest = numData - numTrain - numVal;
trainData = data(1:numTrain, :);
valData = data(numTrain+1:numTrain+numVal, :);
testData = data(numTrain+numVal+1:end, :);

% Define the MLP architecture
numInputNeurons = 6; % 6 inputs
numHiddenNeurons = 6; % Starting with 5 neurons in the hidden layer
numOutputNeurons = 2;

% Define the learning rate and the number of epochs for training
learningRate = 0.000001;
numEpochs = 50000;
bias = 1;

% Initialize variables to store the best weights and validation error
bestWeightsInputToHidden = [];
bestWeightsHiddenToOutput = [];
bestValidationError = inf;
%weight1Record = zeros(numTrain*numEpochs,(numInputNeurons+1)*numHiddenNeurons);
%weight2Record = zeros(numTrain*numEpochs,(numHiddenNeurons+1)*numOutputNeurons);
valLoss = zeros(numEpochs,1);
trainLosses = zeros(numEpochs,1);

% Train the MLP with different numbers of neurons in hidden layer
for numHiddenNeurons = numHiddenNeurons:numHiddenNeurons
    % Initialize the weights for the MLP
    weightsInputToHidden = randn(numInputNeurons + 1, numHiddenNeurons);
    weightsHiddenToOutput = randn(numHiddenNeurons + 1, numOutputNeurons);

    % Train the MLP using backpropagation
    for epoch = 1:numEpochs
        % Shuffle the training data
        shuffledIndices = randperm(numTrain);
        shuffledData = trainData(shuffledIndices, :);

        % Forward pass through the MLP
        input = [bias .* ones(numTrain,1), shuffledData(:, 1:numInputNeurons)]; % Add bias term
        hidden = tanh(input * weightsInputToHidden); % Activation function
        output = tanh([bias .* ones(numTrain,1), hidden] * weightsHiddenToOutput); % Activation function 

        % Update the weights using backward pass through the MLP
        %===============================================================================================================================
        delta = (shuffledData(:, numInputNeurons+2:numInputNeurons+3) - output) .* (1 - output.^2);
        deltas = (delta * weightsHiddenToOutput(2:end,:)') .* (1 - hidden.^2);
        
        weightsHiddenToOutput = weightsHiddenToOutput + [bias .* ones(numTrain,1), hidden]' * (learningRate * delta);
        weightsInputToHidden = weightsInputToHidden + (input' * (learningRate * deltas));
        
        %weight1Record(i+(epoch-1)*numTrain,:) = reshape(weightsInputToHidden.',1,[]);
        %weight2Record(i+(epoch-1)*numTrain,:) = reshape(weightsHiddenToOutput.',1,[]);

        % Compute the training error
        t = shuffledData(:, numInputNeurons+2:end) - output;
        trainLoss = sum(sum((t).^2))/numTrain;
        trainLosses(epoch,:) = trainLoss;

        % Compute the validation error
        input = [bias .* ones(numVal, 1), valData(:, 1:numInputNeurons)]; % Add bias term
        hidden = tanh(input * weightsInputToHidden); % Activation function
        output = tanh([bias .* ones(numVal,1), hidden] * weightsHiddenToOutput); % Activation function  
        t = valData(:, numInputNeurons+2:end) - output;
        validationLoss = sum(sum((t).^2))/numVal;
        valLoss(epoch,:) = validationLoss;
        if mod(epoch, 100) == 0
            fprintf('Epoch %d, Loss: %f\n', epoch, validationLoss);
        end

        % Store the best weights and validation error
        if validationLoss < bestValidationError
            bestWeightsInputToHidden = weightsInputToHidden;
            bestWeightsHiddenToOutput = weightsHiddenToOutput;
            bestValidationError = validationLoss;
        elseif validationLoss > bestValidationError
            %break;
        end
    end
%{
    r = 1:numTrain*numEpochs;


    figure(1);
    subplot(2,1,1);
    plot(r, weight1Record, r, repmat(reshape(bestWeightsInputToHidden.',1,[]), numTrain*numEpochs, 1));
    title("Weights from Input to Hidden vs Iteration");
    subplot(2,1,2);
    plot(r, weight2Record, r, repmat(reshape(bestWeightsHiddenToOutput.',1,[]), numTrain*numEpochs, 1));
    title("Weights from Hidden to Output vs Iteration");
%}

    figure(3);
    plot(1:numEpochs, trainLosses, 'yellow', 1:numEpochs, valLoss, 'blue');
    title("Train Loss(Yellow) and Validation Loss(Blue) vs Number of Epoch");
    legend('Train Loss','Validation Loss');

%{
    % Stop increasing the number of hidden neurons if there is no change in validation error
    if numHiddenNeurons > 5 && validationError >= prevValidationError
        break;
    end
    prevValidationError = validationError;
%}
end

% Evaluate the MLP on the testing set
input = [bias .* ones(numTest, 1), testData(:, 1:numInputNeurons)]; % Add bias term
hidden = tanh(input * bestWeightsInputToHidden); % Activation function
output = tanh([bias .* ones(numVal,1), hidden] * bestWeightsHiddenToOutput); % Activation function  

figure(4);
subplot(2,1,1);
plot(1:numTest, testData(:,end-1), 'green', 1:numTest, output(:,1), 'blue')
subplot(2,1,2);
plot(1:numTest, testData(:,end), 'green', 1:numTest, output(:,2), 'blue')
title("Test set prediction");

%testError = sqrt(mean(sum(testData(:, numInputNeurons+2:end) - output, 2).^2));
%testError = mean(rms(testData(:, numInputNeurons+2:end) - output, 2));

%fprintf('Test Error: %d\n', testError);
toc

