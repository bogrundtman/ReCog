
%**Modifyable values**
%-------------------------------------------------------------------------
input = bodyfatInputs; %set the input source, should be the size of: (nr_of_input_values x nr_of_samples)
target = (bodyfatTargets/100); %set the target source, should be the size of: (nr_of_target_values x nr_of_samples)

%input = transpose(assignment5(:, 2:end));
%target = transpose(assignment5(:, 1))/100;

%adapt the number of layers and nodes in the network.
%Changing the amount of indexes affects nr of layers, changing the value changes nr of nodes.
%First index is input layer, last index is output layer. 
%layer_sizes = [size(input,1), 26, 13, 1];
layer_sizes = [size(input,1), 10, 10, 1]; 

mode = 0; %training or classification mode 0 = training, 1 = classification

%set distribution of data for training.
procent_training = 0.7;
procent_validation = 0.1;
procent_test = 0.2;
%------------------------------------------------------------------------

epoch = 0;


total_n_layers = size(layer_sizes,2);
input_output = cell(total_n_layers,1); %list of matrixes containing the output and input for each layer, including the final output
weights = cell(total_n_layers-1,1); %list of matrixes containing the weights for the layers



if mode == 0 %if network mode is training
    %devide data based on previously set procentages:
    %input data
    [rows, columns] = size(input);
    last_training = int32(floor(procent_training*columns));
    last_validation = int32(floor(procent_validation*columns));
    
    training_input = input(:, 1:last_training);
    input = input(:, last_training+1:end);
    
    validation_input = input(:, 1:last_validation);
    test_input = input(:, last_validation+1:end);

    %target data
    [rows, columns] = size(target);
    last_training = int32(floor(procent_training*columns));
    last_validation = int32(floor(procent_validation*columns));
    
    training_target = target(:, 1:last_training);
    target = target(:, last_training+1:end);
    
    validation_target = target(:, 1:last_validation);
    test_target = target(:, last_validation+1:end);
    
    
    %initilize the weights based on the number of layers
    for i = 1:total_n_layers-1
        weights{i} = initializeWeights(layer_sizes(i), layer_sizes(i+1));
    end
   
    
    while epoch < 10
        
        %training phase
        for i = 1:last_training %loop through each data point
            input_output{1} = training_input(:,i);
            results = forward_prop(input_output, weights, total_n_layers);
            weights = back_prop(results, weights, total_n_layers, training_target(:,i));
        end
       
        
        
        
        %test phase
        input_output{1} = test_input;
        results = forward_prop(input_output, weights, total_n_layers);
        percentage_right_test = mean(ismember(results{total_n_layers}, test_target)) * 100;
        
        %validation phase
        
        input_output{1} = validation_input;
        results = forward_prop(input_output, weights, total_n_layers);
        percentage_right_validation = mean(ismember(results{total_n_layers}, validation_target)) * 100;
        
    
    end
    % Convert cell to a table and use first row as variable names
    T = cell2table(weights);
 
    % Write the table to a CSV file
    writetable(T,'trained_network.csv');
  
    
else %the network is in classification mode
    weights = table2cell(readtable('trained_network.csv'));
    input_output{1} = input;
    results = forward_prop(input_output, weights, total_n_layers);
    output = results{total_n_layers};
end



function input_output = forward_prop(input_output, weights, total_n_layers) %forward propegation function. Currently only uses sigmoid.
    for i = 1:total_n_layers-1
        z = weights{i}*input_output{i}; %multiply input with weights
        input_output{i+1} = sigmoid(z);
    end
end

function weights_updated = back_prop(input_output, weights, total_n_layers, target) %back propegation function. 
    weights_updated = cell(size(weights)); %pre-allocate memory
    error_terms = cell(total_n_layers,1); % pre-allocate memory
    error_terms{1} = target - input_output{total_n_layers}; %calculate error term for output layer
    for i = 1:total_n_layers-1 %calculate error term for the rest of the layers
       
        a = transpose(transpose(error_terms{i}) * weights{(total_n_layers) - i});
        b = sigmoid_derivitive(input_output{(total_n_layers) - i});
        c = a.*b;
        error_terms{i+1} = c;
        
    end
    for i = 1:total_n_layers-1 %calculate change in weight and update weights
        test_d = transpose(input_output{(total_n_layers)-i});
        test_c = error_terms{i};
        deltaW = 0.1 * test_c * test_d;
        t = weights{(size(weights)+1)-i};
        weights_updated{i} = t + deltaW;
    end
    
    weights_updated = flip(weights_updated); %making sure the weights are in the right order :D

end


function weights = initializeWeights(inSize, outSize) %creates random weights based on the size given
  epsilon = 0.12;
  weights = rand(outSize, inSize) * 2 * epsilon - epsilon;
end

function hidden_output = hidden_layer(in, weights)%part of the f
    z = weights*in; %multiply input with weights
    hidden_output = sigmoid(z);
end

function g = sigmoid(z)
    % Computes the sigmoid of z.
    g = 1.0 ./ (1.0 + exp(-z));
end

function g = sigmoid_derivitive(z)
    % Computes the sigmoid derivitive of z.
    x = 1.0 ./ (1.0 + exp(-z));
    g = (exp(-x))./((1+exp(-x)).^2);
    
end

function g = leakyReLU(x)
    
end

    