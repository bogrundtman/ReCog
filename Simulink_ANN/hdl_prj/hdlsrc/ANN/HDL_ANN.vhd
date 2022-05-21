-- -------------------------------------------------------------
-- 
-- File Name: hdl_prj\hdlsrc\ANN\HDL_ANN.vhd
-- Created: 2022-05-21 11:54:13
-- 
-- Generated by MATLAB 9.11 and HDL Coder 3.19
-- 
-- 
-- -------------------------------------------------------------
-- Rate and Clocking Details
-- -------------------------------------------------------------
-- Model base rate: 1
-- Target subsystem base rate: 1
-- 
-- -------------------------------------------------------------


-- -------------------------------------------------------------
-- 
-- Module: HDL_ANN
-- Source Path: ANN/HDL_ANN
-- Hierarchy Level: 0
-- 
-- Simulink model description for ANN:
-- 
-- Symmetric FIR Filter
-- This example shows how to use HDL Coder(TM) to check, generate,
-- and verify HDL for a fixed-point symmetric FIR filter. 
-- 
-- -------------------------------------------------------------
LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;
USE work.HDL_ANN_pkg.ALL;

ENTITY HDL_ANN IS
  PORT( train_1                           :   IN    std_logic;
        input_0                           :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        input_1                           :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        target                            :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        alpha                             :   IN    std_logic_vector(17 DOWNTO 0);  -- ufix18_En18
        w1_0                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_1                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_2                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_3                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_4                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_5                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_6                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_7                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_0                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_1                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_2                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_3                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_4                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_5                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_6                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_7                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w3_0                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w3_1                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b1_0                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b1_1                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b1_2                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b1_3                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b2_0                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b2_1                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b3                                :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        output                            :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_out_0                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_out_1                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_out_2                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_out_3                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_out_4                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_out_5                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_out_6                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w1_out_7                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_out_0                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_out_1                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_out_2                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_out_3                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_out_4                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_out_5                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_out_6                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w2_out_7                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w3_out_0                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        w3_out_1                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b1_out_0                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b1_out_1                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b1_out_2                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b1_out_3                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b2_out_0                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b2_out_1                          :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        b3_out                            :   OUT   std_logic_vector(17 DOWNTO 0)  -- sfix18_En12
        );
END HDL_ANN;


ARCHITECTURE rtl OF HDL_ANN IS

  -- Component Declarations
  COMPONENT Classify
    PORT( x                               :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          w1                              :   IN    matrix_of_std_logic_vector18(0 TO 3, 0 TO 1);  -- sfix18_En12 [4x2]
          w2                              :   IN    matrix_of_std_logic_vector18(0 TO 1, 0 TO 3);  -- sfix18_En12 [2x4]
          w3                              :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          b1                              :   IN    vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
          b2                              :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          b3                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
          a1                              :   OUT   vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
          a2                              :   OUT   vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          a3                              :   OUT   std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
          z1                              :   OUT   vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
          z2                              :   OUT   vector_of_std_logic_vector18(0 TO 1)  -- sfix18_En12 [2]
          );
  END COMPONENT;

  COMPONENT Train
    PORT( x                               :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          y                               :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
          a1                              :   IN    vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
          a2                              :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          a3                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
          z1                              :   IN    vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
          z2                              :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          w1                              :   IN    matrix_of_std_logic_vector18(0 TO 3, 0 TO 1);  -- sfix18_En12 [4x2]
          w2                              :   IN    matrix_of_std_logic_vector18(0 TO 1, 0 TO 3);  -- sfix18_En12 [2x4]
          w3                              :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          b1                              :   IN    vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
          b2                              :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          b3                              :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
          alpha                           :   IN    std_logic_vector(17 DOWNTO 0);  -- ufix18_En18
          Enable                          :   IN    std_logic;
          updated_w1                      :   OUT   matrix_of_std_logic_vector18(0 TO 3, 0 TO 1);  -- sfix18_En12 [4x2]
          updated_w2                      :   OUT   matrix_of_std_logic_vector18(0 TO 1, 0 TO 3);  -- sfix18_En12 [2x4]
          updated_w3                      :   OUT   vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          updated_b1                      :   OUT   vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
          updated_b2                      :   OUT   vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          updated_b3                      :   OUT   std_logic_vector(17 DOWNTO 0)  -- sfix18_En12
          );
  END COMPONENT;

  -- Component Configuration Statements
  FOR ALL : Classify
    USE ENTITY work.Classify(rtl);

  FOR ALL : Train
    USE ENTITY work.Train(rtl);

  -- Signals
  SIGNAL input                            : vector_of_std_logic_vector18(0 TO 1);  -- ufix18 [2]
  SIGNAL w1                               : vector_of_std_logic_vector18(0 TO 7);  -- ufix18 [8]
  SIGNAL w1_8                             : vector_of_signed18(0 TO 7);  -- sfix18_En12 [8]
  SIGNAL Reshape_out1                     : matrix_of_signed18(0 TO 3, 0 TO 1);  -- sfix18_En12 [4x2]
  SIGNAL Reshape_out1_1                   : matrix_of_std_logic_vector18(0 TO 3, 0 TO 1);  -- ufix18 [4x2]
  SIGNAL w2                               : vector_of_std_logic_vector18(0 TO 7);  -- ufix18 [8]
  SIGNAL w2_8                             : vector_of_signed18(0 TO 7);  -- sfix18_En12 [8]
  SIGNAL Reshape1_out1                    : matrix_of_signed18(0 TO 1, 0 TO 3);  -- sfix18_En12 [2x4]
  SIGNAL Reshape1_out1_1                  : matrix_of_std_logic_vector18(0 TO 1, 0 TO 3);  -- ufix18 [2x4]
  SIGNAL w3                               : vector_of_std_logic_vector18(0 TO 1);  -- ufix18 [2]
  SIGNAL b1                               : vector_of_std_logic_vector18(0 TO 3);  -- ufix18 [4]
  SIGNAL b2                               : vector_of_std_logic_vector18(0 TO 1);  -- ufix18 [2]
  SIGNAL Classify_out1                    : vector_of_std_logic_vector18(0 TO 3);  -- ufix18 [4]
  SIGNAL Classify_out2                    : vector_of_std_logic_vector18(0 TO 1);  -- ufix18 [2]
  SIGNAL Classify_out3                    : std_logic_vector(17 DOWNTO 0);  -- ufix18
  SIGNAL Classify_out4                    : vector_of_std_logic_vector18(0 TO 3);  -- ufix18 [4]
  SIGNAL Classify_out5                    : vector_of_std_logic_vector18(0 TO 1);  -- ufix18 [2]
  SIGNAL Reshape_out1_2                   : matrix_of_std_logic_vector18(0 TO 3, 0 TO 1);  -- ufix18 [4x2]
  SIGNAL Reshape1_out1_2                  : matrix_of_std_logic_vector18(0 TO 1, 0 TO 3);  -- ufix18 [2x4]
  SIGNAL Train_out1                       : matrix_of_std_logic_vector18(0 TO 3, 0 TO 1);  -- ufix18 [4x2]
  SIGNAL Train_out2                       : matrix_of_std_logic_vector18(0 TO 1, 0 TO 3);  -- ufix18 [2x4]
  SIGNAL Train_out3                       : vector_of_std_logic_vector18(0 TO 1);  -- ufix18 [2]
  SIGNAL Train_out4                       : vector_of_std_logic_vector18(0 TO 3);  -- ufix18 [4]
  SIGNAL Train_out5                       : vector_of_std_logic_vector18(0 TO 1);  -- ufix18 [2]
  SIGNAL Train_out6                       : std_logic_vector(17 DOWNTO 0);  -- ufix18
  SIGNAL Train_out1_signed                : matrix_of_signed18(0 TO 3, 0 TO 1);  -- sfix18_En12 [4x2]
  SIGNAL Reshape7_out1                    : vector_of_signed18(0 TO 7);  -- sfix18_En12 [8]
  SIGNAL Train_out2_signed                : matrix_of_signed18(0 TO 1, 0 TO 3);  -- sfix18_En12 [2x4]
  SIGNAL Reshape6_out1                    : vector_of_signed18(0 TO 7);  -- sfix18_En12 [8]

BEGIN
  u_Classify : Classify
    PORT MAP( x => input,  -- sfix18_En12 [2]
              w1 => Reshape_out1_1,  -- sfix18_En12 [4x2]
              w2 => Reshape1_out1_1,  -- sfix18_En12 [2x4]
              w3 => w3,  -- sfix18_En12 [2]
              b1 => b1,  -- sfix18_En12 [4]
              b2 => b2,  -- sfix18_En12 [2]
              b3 => b3,  -- sfix18_En12
              a1 => Classify_out1,  -- sfix18_En12 [4]
              a2 => Classify_out2,  -- sfix18_En12 [2]
              a3 => Classify_out3,  -- sfix18_En12
              z1 => Classify_out4,  -- sfix18_En12 [4]
              z2 => Classify_out5  -- sfix18_En12 [2]
              );

  u_Train : Train
    PORT MAP( x => input,  -- sfix18_En12 [2]
              y => target,  -- sfix18_En12
              a1 => Classify_out1,  -- sfix18_En12 [4]
              a2 => Classify_out2,  -- sfix18_En12 [2]
              a3 => Classify_out3,  -- sfix18_En12
              z1 => Classify_out4,  -- sfix18_En12 [4]
              z2 => Classify_out5,  -- sfix18_En12 [2]
              w1 => Reshape_out1_2,  -- sfix18_En12 [4x2]
              w2 => Reshape1_out1_2,  -- sfix18_En12 [2x4]
              w3 => w3,  -- sfix18_En12 [2]
              b1 => b1,  -- sfix18_En12 [4]
              b2 => b2,  -- sfix18_En12 [2]
              b3 => b3,  -- sfix18_En12
              alpha => alpha,  -- ufix18_En18
              Enable => train_1,
              updated_w1 => Train_out1,  -- sfix18_En12 [4x2]
              updated_w2 => Train_out2,  -- sfix18_En12 [2x4]
              updated_w3 => Train_out3,  -- sfix18_En12 [2]
              updated_b1 => Train_out4,  -- sfix18_En12 [4]
              updated_b2 => Train_out5,  -- sfix18_En12 [2]
              updated_b3 => Train_out6  -- sfix18_En12
              );

  input(0) <= input_0;
  input(1) <= input_1;

  w1(0) <= w1_0;
  w1(1) <= w1_1;
  w1(2) <= w1_2;
  w1(3) <= w1_3;
  w1(4) <= w1_4;
  w1(5) <= w1_5;
  w1(6) <= w1_6;
  w1(7) <= w1_7;

  outputgen13: FOR k IN 0 TO 7 GENERATE
    w1_8(k) <= signed(w1(k));
  END GENERATE;

  Reshape_out1GEN_LABEL1: FOR d1 IN 0 TO 1 GENERATE
    Reshape_out1GEN_LABEL: FOR d0 IN 0 TO 3 GENERATE
      Reshape_out1(d0, d1) <= w1_8(d0 + (d1 * 4));
    END GENERATE;
  END GENERATE;

  outputgen11: FOR k IN 0 TO 3 GENERATE
    outputgen12: FOR k1 IN 0 TO 1 GENERATE
      Reshape_out1_1(k, k1) <= std_logic_vector(Reshape_out1(k, k1));
    END GENERATE;
  END GENERATE;

  w2(0) <= w2_0;
  w2(1) <= w2_1;
  w2(2) <= w2_2;
  w2(3) <= w2_3;
  w2(4) <= w2_4;
  w2(5) <= w2_5;
  w2(6) <= w2_6;
  w2(7) <= w2_7;

  outputgen10: FOR k IN 0 TO 7 GENERATE
    w2_8(k) <= signed(w2(k));
  END GENERATE;

  Reshape1_out1GEN_LABEL1: FOR d1 IN 0 TO 3 GENERATE
    Reshape1_out1GEN_LABEL: FOR d0 IN 0 TO 1 GENERATE
      Reshape1_out1(d0, d1) <= w2_8(d0 + (d1 * 2));
    END GENERATE;
  END GENERATE;

  outputgen8: FOR k IN 0 TO 1 GENERATE
    outputgen9: FOR k1 IN 0 TO 3 GENERATE
      Reshape1_out1_1(k, k1) <= std_logic_vector(Reshape1_out1(k, k1));
    END GENERATE;
  END GENERATE;

  w3(0) <= w3_0;
  w3(1) <= w3_1;

  b1(0) <= b1_0;
  b1(1) <= b1_1;
  b1(2) <= b1_2;
  b1(3) <= b1_3;

  b2(0) <= b2_0;
  b2(1) <= b2_1;

  outputgen6: FOR k IN 0 TO 3 GENERATE
    outputgen7: FOR k1 IN 0 TO 1 GENERATE
      Reshape_out1_2(k, k1) <= std_logic_vector(Reshape_out1(k, k1));
    END GENERATE;
  END GENERATE;

  outputgen4: FOR k IN 0 TO 1 GENERATE
    outputgen5: FOR k1 IN 0 TO 3 GENERATE
      Reshape1_out1_2(k, k1) <= std_logic_vector(Reshape1_out1(k, k1));
    END GENERATE;
  END GENERATE;

  outputgen2: FOR k IN 0 TO 3 GENERATE
    outputgen3: FOR k1 IN 0 TO 1 GENERATE
      Train_out1_signed(k, k1) <= signed(Train_out1(k, k1));
    END GENERATE;
  END GENERATE;

  Reshape7_out1GEN_LABEL1: FOR d1 IN 0 TO 1 GENERATE
    Reshape7_out1GEN_LABEL: FOR d0 IN 0 TO 3 GENERATE
      Reshape7_out1(d0 + (d1 * 4)) <= Train_out1_signed(d0, d1);
    END GENERATE;
  END GENERATE;

  w1_out_0 <= std_logic_vector(Reshape7_out1(0));

  w1_out_1 <= std_logic_vector(Reshape7_out1(1));

  w1_out_2 <= std_logic_vector(Reshape7_out1(2));

  w1_out_3 <= std_logic_vector(Reshape7_out1(3));

  w1_out_4 <= std_logic_vector(Reshape7_out1(4));

  w1_out_5 <= std_logic_vector(Reshape7_out1(5));

  w1_out_6 <= std_logic_vector(Reshape7_out1(6));

  w1_out_7 <= std_logic_vector(Reshape7_out1(7));

  outputgen: FOR k IN 0 TO 1 GENERATE
    outputgen1: FOR k1 IN 0 TO 3 GENERATE
      Train_out2_signed(k, k1) <= signed(Train_out2(k, k1));
    END GENERATE;
  END GENERATE;

  Reshape6_out1GEN_LABEL1: FOR d1 IN 0 TO 3 GENERATE
    Reshape6_out1GEN_LABEL: FOR d0 IN 0 TO 1 GENERATE
      Reshape6_out1(d0 + (d1 * 2)) <= Train_out2_signed(d0, d1);
    END GENERATE;
  END GENERATE;

  w2_out_0 <= std_logic_vector(Reshape6_out1(0));

  w2_out_1 <= std_logic_vector(Reshape6_out1(1));

  w2_out_2 <= std_logic_vector(Reshape6_out1(2));

  w2_out_3 <= std_logic_vector(Reshape6_out1(3));

  w2_out_4 <= std_logic_vector(Reshape6_out1(4));

  w2_out_5 <= std_logic_vector(Reshape6_out1(5));

  w2_out_6 <= std_logic_vector(Reshape6_out1(6));

  w2_out_7 <= std_logic_vector(Reshape6_out1(7));

  output <= Classify_out3;

  w3_out_0 <= Train_out3(0);

  w3_out_1 <= Train_out3(1);

  b1_out_0 <= Train_out4(0);

  b1_out_1 <= Train_out4(1);

  b1_out_2 <= Train_out4(2);

  b1_out_3 <= Train_out4(3);

  b2_out_0 <= Train_out5(0);

  b2_out_1 <= Train_out5(1);

  b3_out <= Train_out6;

END rtl;

