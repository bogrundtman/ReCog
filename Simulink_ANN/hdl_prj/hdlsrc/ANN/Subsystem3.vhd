-- -------------------------------------------------------------
-- 
-- File Name: hdl_prj\hdlsrc\ANN\Subsystem3.vhd
-- Created: 2022-05-13 14:13:53
-- 
-- Generated by MATLAB 9.11 and HDL Coder 3.19
-- 
-- -------------------------------------------------------------


-- -------------------------------------------------------------
-- 
-- Module: Subsystem3
-- Source Path: ANN/HDL_ANN/Enabled Subsystem1/Subsystem3
-- Hierarchy Level: 2
-- 
-- -------------------------------------------------------------
LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;
USE work.HDL_ANN_pkg.ALL;

ENTITY Subsystem3 IS
  PORT( w2                                :   IN    matrix_of_std_logic_vector18(0 TO 1, 0 TO 3);  -- sfix18_En12 [2x4]
        a1                                :   IN    vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
        e2                                :   IN    vector_of_std_logic_vector37(0 TO 1);  -- sfix37_En24 [2]
        b2                                :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
        alpha                             :   IN    std_logic_vector(17 DOWNTO 0);  -- ufix18_En18
        updated_b2                        :   OUT   vector_of_std_logic_vector56(0 TO 1);  -- sfix56_En42 [2]
        updated_w2                        :   OUT   matrix_of_std_logic_vector74(0 TO 1, 0 TO 3)  -- sfix74_En54 [2x4]
        );
END Subsystem3;


ARCHITECTURE rtl OF Subsystem3 IS

  -- Signals
  SIGNAL b2_signed                        : vector_of_signed18(0 TO 1);  -- sfix18_En12 [2]
  SIGNAL e2_signed                        : vector_of_signed37(0 TO 1);  -- sfix37_En24 [2]
  SIGNAL alpha_unsigned                   : unsigned(17 DOWNTO 0);  -- ufix18_En18
  SIGNAL Multiply_Add1_mul_cast           : vector_of_signed19(0 TO 1);  -- sfix19_En18 [2]
  SIGNAL Multiply_Add1_mul_mul_temp       : vector_of_signed56(0 TO 1);  -- sfix56_En42 [2]
  SIGNAL mulOutput                        : vector_of_signed55(0 TO 1);  -- sfix55_En42 [2]
  SIGNAL Multiply_Add1_add_sub_cast       : vector_of_signed56(0 TO 1);  -- sfix56_En42 [2]
  SIGNAL Multiply_Add1_add_sub_cast_1     : vector_of_signed56(0 TO 1);  -- sfix56_En42 [2]
  SIGNAL Multiply_Add1_out1               : vector_of_signed56(0 TO 1);  -- sfix56_En42 [2]
  SIGNAL w2_signed                        : matrix_of_signed18(0 TO 1, 0 TO 3);  -- sfix18_En12 [2x4]
  SIGNAL NetworkInPorts_w2                : vector_of_signed18(0 TO 7);  -- sfix18_En12 [8]
  SIGNAL a1_signed                        : vector_of_signed18(0 TO 3);  -- sfix18_En12 [4]
  SIGNAL Transpose_out1                   : vector_of_signed18(0 TO 3);  -- sfix18_En12 [4]
  SIGNAL selector_out                     : vector_of_std_logic_vector37(0 TO 7);  -- ufix37 [8]
  SIGNAL selector_out_1                   : vector_of_signed37(0 TO 7);  -- sfix37_En24 [8]
  SIGNAL s                                : vector_of_signed37(0 TO 7);  -- sfix37_En24 [8]
  SIGNAL selector_out_2                   : vector_of_signed18(0 TO 7);  -- sfix18_En12 [8]
  SIGNAL s_1                              : vector_of_signed18(0 TO 7);  -- sfix18_En12 [8]
  SIGNAL MMul_dot_product_out             : vector_of_signed55(0 TO 7);  -- sfix55_En36 [8]
  SIGNAL dw3_out1                         : matrix_of_signed55(0 TO 1, 0 TO 3);  -- sfix55_En36 [2x4]
  SIGNAL c17_dw3_out1                     : vector_of_signed55(0 TO 7);  -- sfix55_En36 [8]
  SIGNAL Multiply_Add_mul_cast            : vector_of_signed19(0 TO 7);  -- sfix19_En18 [8]
  SIGNAL Multiply_Add_mul_mul_temp        : vector_of_signed74(0 TO 7);  -- sfix74_En54 [8]
  SIGNAL mulOutput_1                      : vector_of_signed73(0 TO 7);  -- sfix73_En54 [8]
  SIGNAL Multiply_Add_add_sub_cast        : vector_of_signed74(0 TO 7);  -- sfix74_En54 [8]
  SIGNAL Multiply_Add_add_sub_cast_1      : vector_of_signed74(0 TO 7);  -- sfix74_En54 [8]
  SIGNAL c22_Multiply_Add_out1            : vector_of_signed74(0 TO 7);  -- sfix74_En54 [8]
  SIGNAL Multiply_Add_out1                : matrix_of_signed74(0 TO 1, 0 TO 3);  -- sfix74_En54 [2x4]

BEGIN
  outputgen8: FOR k IN 0 TO 1 GENERATE
    b2_signed(k) <= signed(b2(k));
  END GENERATE;

  outputgen7: FOR k IN 0 TO 1 GENERATE
    e2_signed(k) <= signed(e2(k));
  END GENERATE;

  alpha_unsigned <= unsigned(alpha);


  mulOutput_gen: FOR t_0 IN 0 TO 1 GENERATE
    Multiply_Add1_mul_cast(t_0) <= signed(resize(alpha_unsigned, 19));
    Multiply_Add1_mul_mul_temp(t_0) <= e2_signed(t_0) * Multiply_Add1_mul_cast(t_0);
    mulOutput(t_0) <= Multiply_Add1_mul_mul_temp(t_0)(54 DOWNTO 0);
  END GENERATE mulOutput_gen;



  Multiply_Add1_out1_gen: FOR t_01 IN 0 TO 1 GENERATE
    Multiply_Add1_add_sub_cast(t_01) <= resize(b2_signed(t_01) & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0', 56);
    Multiply_Add1_add_sub_cast_1(t_01) <= resize(mulOutput(t_01), 56);
    Multiply_Add1_out1(t_01) <= Multiply_Add1_add_sub_cast(t_01) - Multiply_Add1_add_sub_cast_1(t_01);
  END GENERATE Multiply_Add1_out1_gen;


  outputgen6: FOR k IN 0 TO 1 GENERATE
    updated_b2(k) <= std_logic_vector(Multiply_Add1_out1(k));
  END GENERATE;

  outputgen4: FOR k IN 0 TO 1 GENERATE
    outputgen5: FOR k1 IN 0 TO 3 GENERATE
      w2_signed(k, k1) <= signed(w2(k, k1));
    END GENERATE;
  END GENERATE;

  NetworkInPorts_w2GEN_LABEL1: FOR d1 IN 0 TO 3 GENERATE
    NetworkInPorts_w2GEN_LABEL: FOR d0 IN 0 TO 1 GENERATE
      NetworkInPorts_w2(d0 + (d1 * 2)) <= w2_signed(d0, d1);
    END GENERATE;
  END GENERATE;

  outputgen3: FOR k IN 0 TO 3 GENERATE
    a1_signed(k) <= signed(a1(k));
  END GENERATE;

  Transpose_out1 <= a1_signed;

  selector_out(0) <= e2(0);
  selector_out(1) <= e2(1);
  selector_out(2) <= e2(0);
  selector_out(3) <= e2(1);
  selector_out(4) <= e2(0);
  selector_out(5) <= e2(1);
  selector_out(6) <= e2(0);
  selector_out(7) <= e2(1);

  outputgen2: FOR k IN 0 TO 7 GENERATE
    selector_out_1(k) <= signed(selector_out(k));
  END GENERATE;

  sGEN_LABEL: FOR d0 IN 0 TO 7 GENERATE
    s(d0) <= selector_out_1(d0);
  END GENERATE;

  selector_out_2(0) <= Transpose_out1(0);
  selector_out_2(1) <= Transpose_out1(0);
  selector_out_2(2) <= Transpose_out1(1);
  selector_out_2(3) <= Transpose_out1(1);
  selector_out_2(4) <= Transpose_out1(2);
  selector_out_2(5) <= Transpose_out1(2);
  selector_out_2(6) <= Transpose_out1(3);
  selector_out_2(7) <= Transpose_out1(3);

  s_1GEN_LABEL: FOR d0 IN 0 TO 7 GENERATE
    s_1(d0) <= selector_out_2(d0);
  END GENERATE;


  MMul_dot_product_out_gen: FOR t_02 IN 0 TO 7 GENERATE
    MMul_dot_product_out(t_02) <= s(t_02) * s_1(t_02);
  END GENERATE MMul_dot_product_out_gen;


  dw3_out1GEN_LABEL1: FOR d1 IN 0 TO 3 GENERATE
    dw3_out1GEN_LABEL: FOR d0 IN 0 TO 1 GENERATE
      dw3_out1(d0, d1) <= MMul_dot_product_out(d0 + (d1 * 2));
    END GENERATE;
  END GENERATE;

  c17_dw3_out1GEN_LABEL1: FOR d1 IN 0 TO 3 GENERATE
    c17_dw3_out1GEN_LABEL: FOR d0 IN 0 TO 1 GENERATE
      c17_dw3_out1(d0 + (d1 * 2)) <= dw3_out1(d0, d1);
    END GENERATE;
  END GENERATE;


  mulOutput_1_gen: FOR t_03 IN 0 TO 7 GENERATE
    Multiply_Add_mul_cast(t_03) <= signed(resize(alpha_unsigned, 19));
    Multiply_Add_mul_mul_temp(t_03) <= Multiply_Add_mul_cast(t_03) * c17_dw3_out1(t_03);
    mulOutput_1(t_03) <= Multiply_Add_mul_mul_temp(t_03)(72 DOWNTO 0);
  END GENERATE mulOutput_1_gen;



  c22_Multiply_Add_out1_gen: FOR t_04 IN 0 TO 7 GENERATE
    Multiply_Add_add_sub_cast(t_04) <= resize(NetworkInPorts_w2(t_04) & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0', 74);
    Multiply_Add_add_sub_cast_1(t_04) <= resize(mulOutput_1(t_04), 74);
    c22_Multiply_Add_out1(t_04) <= Multiply_Add_add_sub_cast(t_04) - Multiply_Add_add_sub_cast_1(t_04);
  END GENERATE c22_Multiply_Add_out1_gen;


  Multiply_Add_out1GEN_LABEL1: FOR d1 IN 0 TO 3 GENERATE
    Multiply_Add_out1GEN_LABEL: FOR d0 IN 0 TO 1 GENERATE
      Multiply_Add_out1(d0, d1) <= c22_Multiply_Add_out1(d0 + (d1 * 2));
    END GENERATE;
  END GENERATE;

  outputgen: FOR k IN 0 TO 1 GENERATE
    outputgen1: FOR k1 IN 0 TO 3 GENERATE
      updated_w2(k, k1) <= std_logic_vector(Multiply_Add_out1(k, k1));
    END GENERATE;
  END GENERATE;

END rtl;

