for i = 1:10000
    epsilon = 0.12;
    weights_11 = randn(8, 1);
    weights_12 = randn(8, 1);
    weights_13 = randn(8, 1);
    weights_14 = randn(8, 1);
    
    weights_21 = randn(4, 1);
    weights_22 = randn(4, 1);
    weights_3 = randn(2, 1);
 
    b_1 = randn(4, 1);
    b_2 = randn(2, 1);
    b_3 = randn(1, 1);
 
    input = randn(8, 1)*100;

    [out_h1,w_h1, out_h2,w_h2, output] = forwardProp2(weights_11,weights_12, weights_13, weights_14, weights_21,weights_22,weights_3, input);
end
