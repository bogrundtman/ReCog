-- -------------------------------------------------------------
-- 
-- File Name: hdl_prj\hdlsrc\ANN\Subsystem5.vhd
-- Created: 2022-05-13 14:13:53
-- 
-- Generated by MATLAB 9.11 and HDL Coder 3.19
-- 
-- -------------------------------------------------------------


-- -------------------------------------------------------------
-- 
-- Module: Subsystem5
-- Source Path: ANN/HDL_ANN/Enabled Subsystem1/Subsystem5
-- Hierarchy Level: 2
-- 
-- -------------------------------------------------------------
LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;
USE work.HDL_ANN_pkg.ALL;

ENTITY Subsystem5 IS
  PORT( w1                                :   IN    matrix_of_std_logic_vector18(0 TO 3, 0 TO 1);  -- sfix18_En12 [4x2]
        x                                 :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
        e1                                :   IN    vector_of_std_logic_vector55(0 TO 3);  -- sfix55_En36 [4]
        b1                                :   IN    vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
        alpha                             :   IN    std_logic_vector(17 DOWNTO 0);  -- ufix18_En18
        updated_b1                        :   OUT   vector_of_std_logic_vector74(0 TO 3);  -- sfix74_En54 [4]
        updated_w1                        :   OUT   matrix_of_std_logic_vector92(0 TO 3, 0 TO 1)  -- sfix92_En66 [4x2]
        );
END Subsystem5;


ARCHITECTURE rtl OF Subsystem5 IS

  -- Signals
  SIGNAL b1_signed                        : vector_of_signed18(0 TO 3);  -- sfix18_En12 [4]
  SIGNAL e1_signed                        : vector_of_signed55(0 TO 3);  -- sfix55_En36 [4]
  SIGNAL alpha_unsigned                   : unsigned(17 DOWNTO 0);  -- ufix18_En18
  SIGNAL Multiply_Add1_mul_cast           : vector_of_signed19(0 TO 3);  -- sfix19_En18 [4]
  SIGNAL Multiply_Add1_mul_mul_temp       : vector_of_signed74(0 TO 3);  -- sfix74_En54 [4]
  SIGNAL mulOutput                        : vector_of_signed73(0 TO 3);  -- sfix73_En54 [4]
  SIGNAL Multiply_Add1_add_sub_cast       : vector_of_signed74(0 TO 3);  -- sfix74_En54 [4]
  SIGNAL Multiply_Add1_add_sub_cast_1     : vector_of_signed74(0 TO 3);  -- sfix74_En54 [4]
  SIGNAL Multiply_Add1_out1               : vector_of_signed74(0 TO 3);  -- sfix74_En54 [4]
  SIGNAL w1_signed                        : matrix_of_signed18(0 TO 3, 0 TO 1);  -- sfix18_En12 [4x2]
  SIGNAL NetworkInPorts_w1                : vector_of_signed18(0 TO 7);  -- sfix18_En12 [8]
  SIGNAL x_signed                         : vector_of_signed18(0 TO 1);  -- sfix18_En12 [2]
  SIGNAL Transpose_out1                   : vector_of_signed18(0 TO 1);  -- sfix18_En12 [2]
  SIGNAL selector_out                     : vector_of_std_logic_vector55(0 TO 7);  -- ufix55 [8]
  SIGNAL selector_out_1                   : vector_of_signed55(0 TO 7);  -- sfix55_En36 [8]
  SIGNAL s                                : vector_of_signed55(0 TO 7);  -- sfix55_En36 [8]
  SIGNAL selector_out_2                   : vector_of_signed18(0 TO 7);  -- sfix18_En12 [8]
  SIGNAL s_1                              : vector_of_signed18(0 TO 7);  -- sfix18_En12 [8]
  SIGNAL MMul_dot_product_out             : vector_of_signed73(0 TO 7);  -- sfix73_En48 [8]
  SIGNAL dw3_out1                         : matrix_of_signed73(0 TO 3, 0 TO 1);  -- sfix73_En48 [4x2]
  SIGNAL c17_dw3_out1                     : vector_of_signed73(0 TO 7);  -- sfix73_En48 [8]
  SIGNAL Multiply_Add_mul_cast            : vector_of_signed19(0 TO 7);  -- sfix19_En18 [8]
  SIGNAL Multiply_Add_mul_mul_temp        : vector_of_signed92(0 TO 7);  -- sfix92_En66 [8]
  SIGNAL mulOutput_1                      : vector_of_signed91(0 TO 7);  -- sfix91_En66 [8]
  SIGNAL Multiply_Add_add_sub_cast        : vector_of_signed92(0 TO 7);  -- sfix92_En66 [8]
  SIGNAL Multiply_Add_add_sub_cast_1      : vector_of_signed92(0 TO 7);  -- sfix92_En66 [8]
  SIGNAL c22_Multiply_Add_out1            : vector_of_signed92(0 TO 7);  -- sfix92_En66 [8]
  SIGNAL Multiply_Add_out1                : matrix_of_signed92(0 TO 3, 0 TO 1);  -- sfix92_En66 [4x2]

BEGIN
  outputgen8: FOR k IN 0 TO 3 GENERATE
    b1_signed(k) <= signed(b1(k));
  END GENERATE;

  outputgen7: FOR k IN 0 TO 3 GENERATE
    e1_signed(k) <= signed(e1(k));
  END GENERATE;

  alpha_unsigned <= unsigned(alpha);


  mulOutput_gen: FOR t_0 IN 0 TO 3 GENERATE
    Multiply_Add1_mul_cast(t_0) <= signed(resize(alpha_unsigned, 19));
    Multiply_Add1_mul_mul_temp(t_0) <= e1_signed(t_0) * Multiply_Add1_mul_cast(t_0);
    mulOutput(t_0) <= Multiply_Add1_mul_mul_temp(t_0)(72 DOWNTO 0);
  END GENERATE mulOutput_gen;



  Multiply_Add1_out1_gen: FOR t_01 IN 0 TO 3 GENERATE
    Multiply_Add1_add_sub_cast(t_01) <= resize(b1_signed(t_01) & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0', 74);
    Multiply_Add1_add_sub_cast_1(t_01) <= resize(mulOutput(t_01), 74);
    Multiply_Add1_out1(t_01) <= Multiply_Add1_add_sub_cast(t_01) - Multiply_Add1_add_sub_cast_1(t_01);
  END GENERATE Multiply_Add1_out1_gen;


  outputgen6: FOR k IN 0 TO 3 GENERATE
    updated_b1(k) <= std_logic_vector(Multiply_Add1_out1(k));
  END GENERATE;

  outputgen4: FOR k IN 0 TO 3 GENERATE
    outputgen5: FOR k1 IN 0 TO 1 GENERATE
      w1_signed(k, k1) <= signed(w1(k, k1));
    END GENERATE;
  END GENERATE;

  NetworkInPorts_w1GEN_LABEL1: FOR d1 IN 0 TO 1 GENERATE
    NetworkInPorts_w1GEN_LABEL: FOR d0 IN 0 TO 3 GENERATE
      NetworkInPorts_w1(d0 + (d1 * 4)) <= w1_signed(d0, d1);
    END GENERATE;
  END GENERATE;

  outputgen3: FOR k IN 0 TO 1 GENERATE
    x_signed(k) <= signed(x(k));
  END GENERATE;

  Transpose_out1 <= x_signed;

  selector_out(0) <= e1(0);
  selector_out(1) <= e1(1);
  selector_out(2) <= e1(2);
  selector_out(3) <= e1(3);
  selector_out(4) <= e1(0);
  selector_out(5) <= e1(1);
  selector_out(6) <= e1(2);
  selector_out(7) <= e1(3);

  outputgen2: FOR k IN 0 TO 7 GENERATE
    selector_out_1(k) <= signed(selector_out(k));
  END GENERATE;

  sGEN_LABEL: FOR d0 IN 0 TO 7 GENERATE
    s(d0) <= selector_out_1(d0);
  END GENERATE;

  selector_out_2(0) <= Transpose_out1(0);
  selector_out_2(1) <= Transpose_out1(0);
  selector_out_2(2) <= Transpose_out1(0);
  selector_out_2(3) <= Transpose_out1(0);
  selector_out_2(4) <= Transpose_out1(1);
  selector_out_2(5) <= Transpose_out1(1);
  selector_out_2(6) <= Transpose_out1(1);
  selector_out_2(7) <= Transpose_out1(1);

  s_1GEN_LABEL: FOR d0 IN 0 TO 7 GENERATE
    s_1(d0) <= selector_out_2(d0);
  END GENERATE;


  MMul_dot_product_out_gen: FOR t_02 IN 0 TO 7 GENERATE
    MMul_dot_product_out(t_02) <= s(t_02) * s_1(t_02);
  END GENERATE MMul_dot_product_out_gen;


  dw3_out1GEN_LABEL1: FOR d1 IN 0 TO 1 GENERATE
    dw3_out1GEN_LABEL: FOR d0 IN 0 TO 3 GENERATE
      dw3_out1(d0, d1) <= MMul_dot_product_out(d0 + (d1 * 4));
    END GENERATE;
  END GENERATE;

  c17_dw3_out1GEN_LABEL1: FOR d1 IN 0 TO 1 GENERATE
    c17_dw3_out1GEN_LABEL: FOR d0 IN 0 TO 3 GENERATE
      c17_dw3_out1(d0 + (d1 * 4)) <= dw3_out1(d0, d1);
    END GENERATE;
  END GENERATE;


  mulOutput_1_gen: FOR t_03 IN 0 TO 7 GENERATE
    Multiply_Add_mul_cast(t_03) <= signed(resize(alpha_unsigned, 19));
    Multiply_Add_mul_mul_temp(t_03) <= Multiply_Add_mul_cast(t_03) * c17_dw3_out1(t_03);
    mulOutput_1(t_03) <= Multiply_Add_mul_mul_temp(t_03)(90 DOWNTO 0);
  END GENERATE mulOutput_1_gen;



  c22_Multiply_Add_out1_gen: FOR t_04 IN 0 TO 7 GENERATE
    Multiply_Add_add_sub_cast(t_04) <= resize(NetworkInPorts_w1(t_04) & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0', 92);
    Multiply_Add_add_sub_cast_1(t_04) <= resize(mulOutput_1(t_04), 92);
    c22_Multiply_Add_out1(t_04) <= Multiply_Add_add_sub_cast(t_04) - Multiply_Add_add_sub_cast_1(t_04);
  END GENERATE c22_Multiply_Add_out1_gen;


  Multiply_Add_out1GEN_LABEL1: FOR d1 IN 0 TO 1 GENERATE
    Multiply_Add_out1GEN_LABEL: FOR d0 IN 0 TO 3 GENERATE
      Multiply_Add_out1(d0, d1) <= c22_Multiply_Add_out1(d0 + (d1 * 4));
    END GENERATE;
  END GENERATE;

  outputgen: FOR k IN 0 TO 3 GENERATE
    outputgen1: FOR k1 IN 0 TO 1 GENERATE
      updated_w1(k, k1) <= std_logic_vector(Multiply_Add_out1(k, k1));
    END GENERATE;
  END GENERATE;

END rtl;

