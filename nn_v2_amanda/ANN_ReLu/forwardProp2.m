function [out_h1, w_h1, out_h2, w_h2, output] = forwardProp2(weights_11,weights_12, weights_13,weights_14, weights_21,weights_22, weights_3, input)
    %Feed-forward function for network with 2 layers
    
     %since hdl coder does not like matrices as input, devide weight
    %matrices into vectors as input and then combine them to one nice
    %matrix.
    weights_1 = zeros(8,4);
    weights_1(:, 1) = weights_11;
    weights_1(:, 2) = weights_12;
    weights_1(:, 3) = weights_13;
    weights_1(:, 4) = weights_14;
    
    weights_2 = zeros(4,2);
    weights_2(:, 1) = weights_21;
    weights_2(:, 2) = weights_22;
    
    out_h1 = weights_1' * input;
    w_h1 = out_h1;
    out_h1 = max(w_h1, 0);
    
    out_h2 = weights_2' * out_h1;
    w_h2 = out_h2;
    out_h2 = max(w_h2, 0);
    
    output = weights_3' * out_h2;
end