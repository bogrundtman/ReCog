%clear
%network with 2 layers: 8 input neurons, 4 neurons in hidden layer 1 ReLu,
%2 neurons in hidden layer 2 ReLu, 1 output neuron (no activation function)
N = 4;
D_in = 2;
H1 = 30;
H2 = 15;
D_out = 1;

x = [0 0; 0 1; 1 0; 1 1]';

y = [0, 1, 1, 0];


w1 = randn(D_in, H1);
w2 = randn(H1, H2);
w3 = randn(H2, D_out);
prev_best_loss = realmax;

for i = 1:5000
    history_w1{i} = w1;
    history_w2{i} = w2;
    history_w3{i} = w3;

    [out_h1, w_h1, out_h2, w_h2, output] = forwardProp2(w1,w2,w3,x);
    
    history_outputs{i} = output;
    
    t1 = output - y;
    t2 = t1.^2;
    t3= sum(t2);
    losses(i) = t3;
    
    if losses(i) > 200
        w1 = randn(D_in, H1);
        w2 = randn(H1, H2);
        w3 = randn(H2, D_out);
        losses(i) = [];
        continue
    end
    
    if losses(i) < prev_best_loss
        best_w1 = w1;
        best_w2 = w2;
        best_w3 = w3;
        prev_best_loss = losses(i);
    end
    [w1, w2, w3] = back_pro_leakyReLu(x, out_h1, w_h1, out_h2, w_h2, output, w1, w2, w3, y);
end

[out_h1, w_h1, out_h2, w_h2, test_output] = forwardProp2(best_w1,best_w2,best_w3,x);
plot(losses(1:1:end))

function [out_h1, w_h1, out_h2, w_h2, output] = forwardProp2(weights_1,weights_2, weights_3, input)
    %Feed-forward functionf for network with 2 layers
    
    out_h1 = weights_1' * input;
    w_h1 = out_h1;
    out_h1 = max(w_h1, 0);
    
    out_h2 = weights_2' * out_h1;
    w_h2 = out_h2;
    out_h2 = max(w_h2, 0);
    
    output = weights_3' * out_h2;
end

function [out_h1, w_h1, out_h2, w_h2, output] = forwardProp(weights_1,weights_2, weights_3, input)
    %Feed-forward functionf for network with 2 layers
    
    out_h1 = weights_1' * input;
    w_h1 = out_h1;
    for i = 1:30 
        if out_h1(i, 1) < 0 %leaky ReLU
            out_h1(i, 1) = 0.01*out_h1(i, 1);
        end
    end
    out_h2 = weights_2' * out_h1;
    w_h2 = out_h2;
    for i = 1:15
        if out_h2(i, 1) < 0 %leaky ReLU
            out_h2(i, 1) = 0.01*out_h2(i, 1);
        end
    end
    output = weights_3' * out_h2;
end

function [w1, w2, w3] = back_pro_leakyReLu(input, out_h1, w_h1, out_h2, w_h2, output, weights_1, weights_2, weights_3, target)
    learning_rate = 0.002;
    
    %output layer
    grad_output = (2.0 * (output - target))';
    grad_w3 =  out_h2 * grad_output;
    
    %hidden layer nr 2
    grad_h2_relu = grad_output * weights_3';
    grad_h2 = grad_h2_relu;
    grad_h2(w_h2 < 0) = 0.0;
    grad_h2(w_h2 > 0) = 1;
    grad_w2 = out_h1 * grad_h2;
    
    %hidden layer nr 1
    grad_h1_relu = grad_h2 * weights_2';
    grad_h1 = grad_h1_relu;
    grad_h1(w_h1 < 0) = 0.0;
    grad_h1(w_h1 > 0) = 1;
    grad_w1 = input * grad_h1;
    
    %updating the weights
    w1 = weights_1 - (learning_rate * grad_w1);
    w2 = weights_2 - (learning_rate * grad_w2);
    w3 = weights_3 - (learning_rate * grad_w3);
    
    
end

function weights = initializeWeights(inSize, outSize) %creates random weights based on the size given
  epsilon = 0.12;
  weights = rand(outSize, inSize) * 2 * epsilon - epsilon;
end




