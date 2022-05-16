-- -------------------------------------------------------------
-- 
-- File Name: hdl_prj\hdlsrc\ANN\Enabled_Subsystem.vhd
-- Created: 2022-05-13 14:13:53
-- 
-- Generated by MATLAB 9.11 and HDL Coder 3.19
-- 
-- -------------------------------------------------------------


-- -------------------------------------------------------------
-- 
-- Module: Enabled_Subsystem
-- Source Path: ANN/HDL_ANN/Enabled Subsystem
-- Hierarchy Level: 1
-- 
-- -------------------------------------------------------------
LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;
USE work.HDL_ANN_pkg.ALL;

ENTITY Enabled_Subsystem IS
  PORT( x                                 :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
        w1                                :   IN    matrix_of_std_logic_vector18(0 TO 3, 0 TO 1);  -- sfix18_En12 [4x2]
        w2                                :   IN    matrix_of_std_logic_vector18(0 TO 1, 0 TO 3);  -- sfix18_En12 [2x4]
        w3                                :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
        b1                                :   IN    vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
        b2                                :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
        b3                                :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
        Enable                            :   IN    std_logic;
        a1                                :   OUT   vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
        a2                                :   OUT   vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
        a3                                :   OUT   std_logic_vector(17 DOWNTO 0)  -- sfix18_En12
        );
END Enabled_Subsystem;


ARCHITECTURE rtl OF Enabled_Subsystem IS

  -- Component Declarations
  COMPONENT L1
    PORT( a                               :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          w                               :   IN    matrix_of_std_logic_vector18(0 TO 3, 0 TO 1);  -- sfix18_En12 [4x2]
          b                               :   IN    vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
          o                               :   OUT   vector_of_std_logic_vector18(0 TO 3)  -- sfix18_En12 [4]
          );
  END COMPONENT;

  COMPONENT L2
    PORT( a                               :   IN    vector_of_std_logic_vector18(0 TO 3);  -- sfix18_En12 [4]
          w                               :   IN    matrix_of_std_logic_vector18(0 TO 1, 0 TO 3);  -- sfix18_En12 [2x4]
          b                               :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          o                               :   OUT   vector_of_std_logic_vector18(0 TO 1)  -- sfix18_En12 [2]
          );
  END COMPONENT;

  COMPONENT L3
    PORT( a                               :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          w                               :   IN    vector_of_std_logic_vector18(0 TO 1);  -- sfix18_En12 [2]
          b                               :   IN    std_logic_vector(17 DOWNTO 0);  -- sfix18_En12
          o                               :   OUT   std_logic_vector(17 DOWNTO 0)  -- sfix18_En12
          );
  END COMPONENT;

  -- Component Configuration Statements
  FOR ALL : L1
    USE ENTITY work.L1(rtl);

  FOR ALL : L2
    USE ENTITY work.L2(rtl);

  FOR ALL : L3
    USE ENTITY work.L3(rtl);

  -- Signals
  SIGNAL L1_out1                          : vector_of_std_logic_vector18(0 TO 3);  -- ufix18 [4]
  SIGNAL L2_out1                          : vector_of_std_logic_vector18(0 TO 1);  -- ufix18 [2]
  SIGNAL L3_out1                          : std_logic_vector(17 DOWNTO 0);  -- ufix18

BEGIN
  u_L1 : L1
    PORT MAP( a => x,  -- sfix18_En12 [2]
              w => w1,  -- sfix18_En12 [4x2]
              b => b1,  -- sfix18_En12 [4]
              o => L1_out1  -- sfix18_En12 [4]
              );

  u_L2 : L2
    PORT MAP( a => L1_out1,  -- sfix18_En12 [4]
              w => w2,  -- sfix18_En12 [2x4]
              b => b2,  -- sfix18_En12 [2]
              o => L2_out1  -- sfix18_En12 [2]
              );

  u_L3 : L3
    PORT MAP( a => L2_out1,  -- sfix18_En12 [2]
              w => w3,  -- sfix18_En12 [2]
              b => b3,  -- sfix18_En12
              o => L3_out1  -- sfix18_En12
              );

  a1 <= L1_out1;

  a2 <= L2_out1;

  a3 <= L3_out1;


END rtl;

