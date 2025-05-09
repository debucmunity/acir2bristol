use acir::circuit::opcodes::*;
use acir::circuit::*;
use acir::native_types::*;
use acir_field::*;
use serde::Serialize;

use std::path::Path;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fs;
use std::fs::*;
use std::io::{BufRead, Read, BufReader, Cursor, Write};

use bristol_circuit::{BristolCircuit, CircuitInfo, Gate, IOInfo};
use serde_json::json;

use boolify::boolify;

fn bristol_translator<F: AcirField>(circuit: &Circuit<F>) -> BristolCircuit {
    // idx in acir -> idx in bristol arithmetic
    let mut wire_translator: HashMap<u32, u32> = HashMap::new();
    let mut total_wires = 0;

    let mut gate: Vec<Vec<u32>> = Vec::new(); // 0 -> AAdd, 1 -> ASub, 2 -> AMul

    for (i, opcode) in circuit.opcodes.iter().enumerate() {
        println!("Opcode {}: {:?}", i, opcode);

        match opcode {
            Opcode::BlackBoxFuncCall(func) => {
                match func {
                    acir::circuit::opcodes::BlackBoxFuncCall::BigIntAdd { lhs, rhs, output } => {
                        // println!("    - BigIntAdd:");
                        // println!("      - LHS: {:?}", lhs);
                        // println!("      - RHS: {:?}", rhs);
                        // println!("      - Output: {:?}", output);

                        let mut temp_vec = Vec::new();

                        if !wire_translator.contains_key(lhs) {
                            wire_translator.insert(lhs.clone(), total_wires);
                            total_wires += 1;
                        }
                        temp_vec.push(wire_translator.get(lhs).copied().unwrap_or(0));

                        if !wire_translator.contains_key(rhs) {
                            wire_translator.insert(rhs.clone(), total_wires);
                            total_wires += 1;
                        }
                        temp_vec.push(wire_translator.get(rhs).copied().unwrap_or(0));

                        if !wire_translator.contains_key(output) {
                            wire_translator.insert(output.clone(), total_wires);
                            total_wires += 1;
                        }
                        temp_vec.push(wire_translator.get(output).copied().unwrap_or(0));

                        temp_vec.push(0);

                        gate.push(temp_vec);
                    }
                    acir::circuit::opcodes::BlackBoxFuncCall::BigIntSub { lhs, rhs, output } => {
                        // println!("    - BigIntSub:");
                        // println!("      - LHS: {:?}", lhs);
                        // println!("      - RHS: {:?}", rhs);
                        // println!("      - Output: {:?}", output);

                        let mut temp_vec = Vec::new();

                        if !wire_translator.contains_key(lhs) {
                            wire_translator.insert(lhs.clone(), total_wires);
                            total_wires += 1;
                        }
                        temp_vec.push(wire_translator.get(lhs).copied().unwrap_or(0));

                        if !wire_translator.contains_key(rhs) {
                            wire_translator.insert(rhs.clone(), total_wires);
                            total_wires += 1;
                        }
                        temp_vec.push(wire_translator.get(rhs).copied().unwrap_or(0));

                        if !wire_translator.contains_key(output) {
                            wire_translator.insert(output.clone(), total_wires);
                            total_wires += 1;
                        }
                        temp_vec.push(wire_translator.get(output).copied().unwrap_or(0));

                        temp_vec.push(1);

                        gate.push(temp_vec);
                    }
                    acir::circuit::opcodes::BlackBoxFuncCall::BigIntMul { lhs, rhs, output } => {
                        // println!("    - BigIntMul:");
                        // println!("      - LHS: {:?}", lhs);
                        // println!("      - RHS: {:?}", rhs);
                        // println!("      - Output: {:?}", output);

                        let mut temp_vec = Vec::new();

                        if !wire_translator.contains_key(lhs) {
                            wire_translator.insert(lhs.clone(), total_wires);
                            total_wires += 1;
                        }
                        temp_vec.push(wire_translator.get(lhs).copied().unwrap_or(0));

                        if !wire_translator.contains_key(rhs) {
                            wire_translator.insert(rhs.clone(), total_wires);
                            total_wires += 1;
                        }
                        temp_vec.push(wire_translator.get(rhs).copied().unwrap_or(0));

                        if !wire_translator.contains_key(output) {
                            wire_translator.insert(output.clone(), total_wires);
                            total_wires += 1;
                        }
                        temp_vec.push(wire_translator.get(output).copied().unwrap_or(0));

                        temp_vec.push(2);

                        gate.push(temp_vec);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    // println!("{:?}", gate);

    let mut inps: Vec<IOInfo> = Vec::new();
    let mut outs: Vec<IOInfo> = Vec::new();
    let mut add_gates: Vec<Gate> = Vec::new();

    let mut count_inp = 0;
    let mut count_out = 0;

    // gate: [input0, input1, output, gate_type]
    for element in gate {
        inps.push(IOInfo {
            name: format!("input{}", count_inp),
            type_: json!("number"),
            address: element[0].try_into().unwrap(),
            width: 1,
        });
        count_inp += 1;

        inps.push(IOInfo {
            name: format!("input{}", count_inp),
            type_: json!("number"),
            address: element[1].try_into().unwrap(),
            width: 1,
        });
        count_inp += 1;

        outs.push(IOInfo {
            name: format!("output{}", count_out),
            type_: json!("number"),
            address: element[2].try_into().unwrap(),
            width: 1,
        });
        count_out += 1;

        let mut op_gate = String::new();
        if element[3] == 0 {
            op_gate = "AAdd".to_string();
        } else if element[3] == 1 {
            op_gate = "ASub".to_string();
        } else if element[3] == 2 {
            op_gate = "AMul".to_string();
        }

        add_gates.push(Gate {
            inputs: vec![
                element[0].try_into().unwrap(),
                element[1].try_into().unwrap(),
            ],
            outputs: vec![element[2].try_into().unwrap()],
            op: op_gate,
        });
    }

    BristolCircuit {
        wire_count: total_wires.try_into().unwrap(),
        info: CircuitInfo {
            constants: vec![],
            inputs: inps,
            outputs: outs,
        },
        gates: add_gates,
    }
}

fn load_circuit_from_json(path: &str) -> Result<Circuit<FieldElement>, Box<dyn std::error::Error>> {
    let json_str = fs::read_to_string(path)?;
    let circuit: Circuit<FieldElement> = serde_json::from_str(&json_str)?;
    Ok(circuit)
}

fn save_circuit_to_json(path: &str, circuit: &Circuit<FieldElement>) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(&circuit)?;
    write(path, json)?;
    Ok(())
}

fn main() ->  Result<(), Box<dyn std::error::Error>>  {
    const BITS_PER_NUM: u32 = 64;

    // Define witness ranges for each operand and result for three operations
    let add_lhs_start = 1;
    let add_rhs_start = add_lhs_start + BITS_PER_NUM;
    let add_out_start = add_rhs_start + BITS_PER_NUM;

    let sub_lhs_start = add_out_start + BITS_PER_NUM;
    let sub_rhs_start = sub_lhs_start + BITS_PER_NUM;
    let sub_out_start = sub_rhs_start + BITS_PER_NUM;

    let mul_lhs_start = sub_out_start + BITS_PER_NUM;
    let mul_rhs_start = mul_lhs_start + BITS_PER_NUM;
    let mul_out_start = mul_rhs_start + BITS_PER_NUM;

    // Calculate the highest witness index that will be used
    let current_max_witness = mul_out_start + BITS_PER_NUM - 1;

    // println!(
    //     "Max witness index defined for 3 operations ({} bits each): {}. Next witness index: {}.",
    //     BITS_PER_NUM,
    //     current_max_witness,
    //     current_max_witness + 1
    // );

    let mut all_public_params = BTreeSet::new();
    // Add all 64-bit inputs for each operation as public parameters
    for i in 0..BITS_PER_NUM {
        all_public_params.insert(Witness(add_lhs_start + i)); // ADD LHS
        all_public_params.insert(Witness(add_rhs_start + i)); // ADD RHS

        all_public_params.insert(Witness(sub_lhs_start + i)); // SUB LHS
        all_public_params.insert(Witness(sub_rhs_start + i)); // SUB RHS

        all_public_params.insert(Witness(mul_lhs_start + i)); // MUL LHS
        all_public_params.insert(Witness(mul_rhs_start + i)); // MUL RHS
    }

    let mut all_return_values = BTreeSet::new();
    // Add all 64-bit outputs for each operation as return values
    for i in 0..BITS_PER_NUM {
        all_return_values.insert(Witness(add_out_start + i)); // ADD Output
        all_return_values.insert(Witness(sub_out_start + i)); // SUB Output
        all_return_values.insert(Witness(mul_out_start + i)); // MUL Output
    }

    let circuit = Circuit::<FieldElement> {
        current_witness_index: current_max_witness + 1, // Set to one greater than the max witness used
        opcodes: vec![
            Opcode::BlackBoxFuncCall(BlackBoxFuncCall::BigIntAdd {
                lhs: add_lhs_start, // Base witness for the 64-bit number
                rhs: add_rhs_start,
                output: add_out_start,
            }),
            Opcode::BlackBoxFuncCall(BlackBoxFuncCall::BigIntSub {
                lhs: sub_lhs_start,
                rhs: sub_rhs_start,
                output: sub_out_start,
            }),
            Opcode::BlackBoxFuncCall(BlackBoxFuncCall::BigIntMul {
                lhs: mul_lhs_start,
                rhs: mul_rhs_start,
                output: mul_out_start,
            }),
        ],
        expression_width: ExpressionWidth::Unbounded,
        private_parameters: BTreeSet::new(), // Assuming all inputs are public for this example
        public_parameters: PublicInputs(all_public_params),
        return_values: PublicInputs(all_return_values),
        assert_messages: Vec::new(),
    };

    println!("\nACIR Circuit:\n{}", circuit.to_string());
    save_circuit_to_json("circuit_acir.json", &circuit)?;
    let circuit2 = load_circuit_from_json("circuit_acir.json")?; 


    let arith_circuit = bristol_translator(&circuit2);

    let circuit_str = arith_circuit.get_bristol_string().unwrap();
    let circuit_info: CircuitInfo = arith_circuit.info;
    println!("{:?}", circuit_info);
    let circuit_info_string = serde_json::to_string(&circuit_info)?;
    let mut file = File::create("circuit_info.txt").unwrap();
    file.write_all(circuit_info_string.as_bytes()).unwrap();

    let new_arith_circuit =
        BristolCircuit::read_info_and_bristol(&circuit_info, &mut Cursor::new(circuit_str.clone()))
            .unwrap();


    let bool_circuit = boolify(&new_arith_circuit, BITS_PER_NUM.try_into().unwrap());

    let file_str = bool_circuit.get_bristol_string().unwrap();
    // println!("{:?}", file_str);

    let mut file = File::create("boolean-Bristol.txt");
    let _ = file.expect("REASON").write_all(file_str.as_bytes());

    let mut arith_file = File::create("arith-Bristol.txt");
    let _ = arith_file
        .expect("REASON")
        .write_all(circuit_str.as_bytes());
    Ok(())
}
