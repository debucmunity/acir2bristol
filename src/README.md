# ACIR Circuit to Bristol (Arithmetic or Boolean) Circuit

Based on the circuit defined in a main function in file `src/main.rs` (line 189 - 260), our project will generate 2 file as `boolean-Bristol.txt` and `arith-Bristol.txt` for further use (e.g. apply this circuit into MPC framework).

## Explanation

- `bristol_translator` function: convert ACIR circuit to arithmetic bristol circuit.

    + Convert 3 opcodes in ACIR as BigIntAdd, BigIntSub, BigIntMul to the associated opcodes in arithmetic Bristol (AAdd, ASub, AMul).

    + Do not handle immediate wire. Our code translate wire type into only input and output type.

- In a main function, from the line 270, apply `boolify` function to convert from arithmetic bristol circuit to boolean bristol circuit.
