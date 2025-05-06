use std::collections::HashMap;
use acir::circuit::opcodes::*;
use acir::circuit::*;
use acir::native_types::*;
use acir_field::*;
use std::collections::BTreeSet;

// Bristol format circuit structures
#[derive(Clone, Debug, PartialEq)]
pub enum WireType {
    Input,
    Output,
    Intermediate,
    Constant,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GateType {
    XOR,
    AND,
    INV,
    EQ,
    EQW,
    MAND,
}

#[derive(Clone, Debug)]
pub struct BristolGate {
    pub gate_type: GateType,
    pub inputs: Vec<usize>,  // Wire indices
    pub output: usize,       // Wire index
}

#[derive(Clone, Debug)]
pub struct BristolCircuit<F: AcirField> {
    pub gates: Vec<BristolGate>,
    pub num_wires: usize,
    pub wire_usage: Vec<F>,
    pub wire_types: Vec<WireType>,
    pub input_values: Vec<Vec<usize>>,  // Groups of input wires
    pub output_values: Vec<Vec<usize>>, // Groups of output wires
}

impl<F: AcirField> BristolCircuit<F> {
    pub fn new() -> Self {
        Self {
            gates: Vec::new(),
            num_wires: 0,
            wire_usage: Vec::new(),
            wire_types: Vec::new(),
            input_values: Vec::new(),
            output_values: Vec::new(),
        }
    }

    pub fn add_input_value(&mut self, num_wires: u32) -> Result<Vec<usize>, String> {
        let mut wires = Vec::new();
        for _ in 0..num_wires {
            let wire = self.num_wires;
            self.num_wires += 1;
            self.wire_usage.push(F::zero());
            self.wire_types.push(WireType::Input);
            wires.push(wire);
        }
        self.input_values.push(wires.clone());
        Ok(wires)
    }

    pub fn add_output_value(&mut self, num_wires: u32) -> Result<Vec<usize>, String> {
        let mut wires = Vec::new();
        for _ in 0..num_wires {
            let wire = self.num_wires;
            self.num_wires += 1;
            self.wire_usage.push(F::zero());
            self.wire_types.push(WireType::Output);
            wires.push(wire);
        }
        self.output_values.push(wires.clone());
        Ok(wires)
    }

    pub fn add_gate(&mut self, gate_type: GateType, inputs: Vec<usize>, output: usize) -> Result<(), String> {
        println!("Adding gate: {:?} with inputs {:?} and output {}", gate_type, inputs, output);
        
        // Validate inputs
        match gate_type {
            GateType::INV | GateType::EQ | GateType::EQW => {
                if inputs.len() != 1 {
                    return Err("INV/EQ/EQW gates must have exactly one input".to_string());
                }
            }
            GateType::XOR | GateType::AND => {
                if inputs.len() != 2 {
                    return Err("XOR/AND gates must have exactly two inputs".to_string());
                }
            }
            GateType::MAND => {
                if inputs.len() < 2 || inputs.len() % 2 != 0 {
                    return Err("MAND gates must have an even number of inputs".to_string());
                }
            }
        }

        // Validate input wires exist and are not outputs
        for &input in &inputs {
            if input >= self.num_wires {
                return Err(format!("Input wire {} does not exist", input));
            }
            if self.wire_types[input] == WireType::Output {
                return Err(format!("Cannot use output wire {} as input", input));
            }
            println!("Input wire {} type: {:?}, current usage: {}", input, self.wire_types[input], self.wire_usage[input]);
        }

        // Validate output wire
        if output >= self.num_wires {
            return Err(format!("Output wire {} does not exist", output));
        }
        println!("Output wire {} type: {:?}, current usage: {}", output, self.wire_types[output], self.wire_usage[output]);
        
        // Update wire usage for inputs
        for &input in &inputs {
            self.wire_usage[input] = self.wire_usage[input] + F::one();
            println!("Updated input wire {} usage to {}", input, self.wire_usage[input]);
        }
        
        // Update wire usage for output
        self.wire_usage[output] = self.wire_usage[output] + F::one();
        println!("Updated output wire {} usage to {}", output, self.wire_usage[output]);
        
        // Add the gate
        self.gates.push(BristolGate {
            gate_type,
            inputs,
            output,
        });
        
        Ok(())
    }

    pub fn to_bristol_format(&self) -> Result<String, String> {
        self.validate()?;
        
        let mut output = String::new();
        
        output.push_str(&format!("{} {}\n", self.gates.len(), self.num_wires));
        
        output.push_str(&format!("{}\n", self.input_values.len()));
        
        for input_value in &self.input_values {
            output.push_str(&format!("{} ", input_value.len()));
        }
        output.push_str("\n");
        
        output.push_str(&format!("{}\n", self.output_values.len()));
        
        // Number of output wires per output value
        for output_value in &self.output_values {
            output.push_str(&format!("{} ", output_value.len()));
        }
        output.push_str("\n");
         
        for gate in &self.gates {
            match gate.gate_type {
                GateType::XOR => {
                    output.push_str(&format!("2 1 {} {} {} XOR\n", 
                        gate.inputs[0], gate.inputs[1], gate.output));
                }
                GateType::AND => {
                    output.push_str(&format!("2 1 {} {} {} AND\n", 
                        gate.inputs[0], gate.inputs[1], gate.output));
                }
                GateType::INV => {
                    output.push_str(&format!("1 1 {} {} INV\n", 
                        gate.inputs[0], gate.output));
                }
                GateType::EQ => {
                    output.push_str(&format!("1 1 {} {} EQ\n", 
                        gate.inputs[0], gate.output));
                }
                GateType::EQW => {
                    output.push_str(&format!("1 1 {} {} EQW\n", 
                        gate.inputs[0], gate.output));
                }
                GateType::MAND => {
                    let num_outputs = gate.inputs.len() / 2;
                    output.push_str(&format!("{} {} ", gate.inputs.len(), num_outputs));
                    for input in &gate.inputs {
                        output.push_str(&format!("{} ", input));
                    }
                    output.push_str(&format!("{} MAND\n", gate.output));
                }
            }
        }
        
        Ok(output)
    }

    pub fn validate(&self) -> Result<(), String> {
        println!("\nValidating circuit:");
        println!("Total wires: {}", self.num_wires);
        println!("Total gates: {}", self.gates.len());
        
        for (i, &usage) in self.wire_usage.iter().enumerate() {
            println!("Wire {}: type {:?}, usage {}", i, self.wire_types[i], usage);
            if usage == F::zero() && self.wire_types[i] != WireType::Output {
                return Err(format!("Wire {} is unused", i));
            }
        }

        let mut visited = vec![false; self.num_wires];
        let mut stack = vec![false; self.num_wires];
        
        for i in 0..self.num_wires {
            if !visited[i] {
                if self.has_cycle(i, &mut visited, &mut stack) {
                    return Err("Circuit contains cycles".to_string());
                }
            }
        }

        Ok(())
    }

    fn has_cycle(&self, wire: usize, visited: &mut [bool], stack: &mut [bool]) -> bool {
        visited[wire] = true;
        stack[wire] = true;

        for gate in &self.gates {
            if gate.output == wire {
                for &input in &gate.inputs {
                    if !visited[input] {
                        if self.has_cycle(input, visited, stack) {
                            return true;
                        }
                    } else if stack[input] {
                        return true;
                    }
                }
            }
        }

        stack[wire] = false;
        false
    }

    pub fn connect_input(&mut self, input_index: usize, wire: usize) -> Result<(), String> {
        println!("Connecting input {} to wire {}", input_index, wire);
        
        if input_index >= self.input_values.len() {
            return Err(format!("Input index {} out of bounds", input_index));
        }
        if wire >= self.num_wires {
            return Err(format!("Wire {} does not exist", wire));
        }
        
        // Update wire usage
        self.wire_usage[wire] = self.wire_usage[wire] + F::one();
        println!("Updated wire {} usage to {}", wire, self.wire_usage[wire]);
        
        Ok(())
    }

    pub fn get_output(&self, output_index: usize) -> Result<usize, String> {
        if output_index >= self.output_values.len() {
            return Err(format!("Output index {} out of bounds", output_index));
        }
        // Find the wire that corresponds to this output
        for (i, wire_type) in self.wire_types.iter().enumerate() {
            if *wire_type == WireType::Output {
                if output_index == 0 {
                    return Ok(i);
                }
            }
        }
        Err(format!("Output wire {} not found", output_index))
    }

    pub fn merge_subcircuit(&mut self, other: BristolCircuit<F>) -> Result<(), String> {
        println!("\nMerging subcircuit:");
        println!("Original circuit: {} wires, {} gates", self.num_wires, self.gates.len());
        println!("Subcircuit: {} wires, {} gates", other.num_wires, other.gates.len());
        
        let mut wire_map = Vec::new();
        for i in 0..other.num_wires {
            let new_wire = self.num_wires;
            self.num_wires += 1;
            self.wire_usage.push(F::zero());
            self.wire_types.push(other.wire_types[i].clone());
            wire_map.push(new_wire);
            println!("Mapped wire {} to {}", i, new_wire);
        }

        for gate in other.gates {
            let mut new_inputs = Vec::new();
            for input in gate.inputs {
                new_inputs.push(wire_map[input]);
            }
            let new_output = wire_map[gate.output];
            
            println!("Adding gate: {:?} with inputs {:?} and output {}", 
                    gate.gate_type, new_inputs, new_output);
            
            self.gates.push(BristolGate {
                gate_type: gate.gate_type,
                inputs: new_inputs,
                output: new_output,
            });
        }

        println!("Merged circuit: {} wires, {} gates", self.num_wires, self.gates.len());
        Ok(())
    }
}

pub struct BristolCircuitTranslator<F: AcirField> {
    pub circuit: BristolCircuit<F>,
    pub witness_wire_map: HashMap<Witness, usize>,
}

impl<F: AcirField> BristolCircuitTranslator<F> {
    pub fn new() -> Self {
        Self {
            circuit: BristolCircuit::new(),
            witness_wire_map: HashMap::new(),
        }
    }

    pub fn translate_circuit(&mut self, circuit: &Circuit<F>) -> Result<(), String> {
        for witness in circuit.public_parameters.0.iter() {
            self.register_witness(*witness)?;
        }
        for witness in circuit.private_parameters.iter() {
            self.register_witness(*witness)?;
        }

        for opcode in &circuit.opcodes {
            self.translate_opcode(opcode)?;
        }

        for witness in circuit.return_values.0.iter() {
            if let Some(wire) = self.witness_wire_map.get(witness) {
                self.circuit.wire_usage[*wire] = self.circuit.wire_usage[*wire] + F::one();
            }
        }

        Ok(())
    }

    fn register_witness(&mut self, witness: Witness) -> Result<(), String> {
        if !self.witness_wire_map.contains_key(&witness) {
            let wires = self.circuit.add_input_value(8)?; // Default to 8 bits
            self.witness_wire_map.insert(witness, wires[0]);
        }
        Ok(())
    }

    fn get_or_create_wire(&mut self, witness: Witness) -> Result<usize, String> {
        match self.witness_wire_map.get(&witness) {
            Some(wire) => Ok(*wire),
            None => {
                let wires = self.circuit.add_input_value(8)?; // Default to 8 bits
                self.witness_wire_map.insert(witness, wires[0]);
                Ok(wires[0])
            }
        }
    }

    fn translate_opcode(&mut self, opcode: &Opcode<F>) -> Result<(), String> {
        match opcode {
            Opcode::AssertZero(expr) => {
                self.translate_expression(expr)?;
            }
            Opcode::BlackBoxFuncCall(func_call) => {
                match func_call {
                    BlackBoxFuncCall::AND { lhs, rhs, output } => {
                        let lhs_wire = self.get_or_create_wire(lhs.to_witness())?;
                        let rhs_wire = self.get_or_create_wire(rhs.to_witness())?;
                        let output_wires = self.circuit.add_output_value(1)?;
                        let _ = self.circuit.add_gate(
                            GateType::AND,
                            vec![lhs_wire, rhs_wire],
                            output_wires[0],
                        )?;
                        self.witness_wire_map.insert(*output, output_wires[0]);
                    }
                    BlackBoxFuncCall::XOR { lhs, rhs, output } => {
                        let lhs_wire = self.get_or_create_wire(lhs.to_witness())?;
                        let rhs_wire = self.get_or_create_wire(rhs.to_witness())?;
                        let output_wires = self.circuit.add_output_value(1)?;
                        let _ = self.circuit.add_gate(
                            GateType::XOR,
                            vec![lhs_wire, rhs_wire],
                            output_wires[0],
                        )?;
                        self.witness_wire_map.insert(*output, output_wires[0]);
                    }
                    BlackBoxFuncCall::RANGE { input } => {
                        let input_wire = self.get_or_create_wire(input.to_witness())?;
                        let constant_wires = self.circuit.add_input_value(input.num_bits())?;
                        let output_wires = self.circuit.add_output_value(1)?;
                        let _ = self.circuit.add_gate(
                            GateType::AND,
                            vec![input_wire, constant_wires[0]],
                            output_wires[0],
                        )?;
                    }
                    BlackBoxFuncCall::Sha256Compression { inputs, hash_values, outputs } => {
                        let mut sha256_circuit = self.load_sha256_subcircuit()?;
                        
                        for (i, input) in inputs.iter().enumerate() {
                            let input_wire = self.get_or_create_wire(input.to_witness())?;
                            sha256_circuit.connect_input(i, input_wire)?;
                        }
                        
                        for (i, hash_value) in hash_values.iter().enumerate() {
                            let hash_wire = self.get_or_create_wire(hash_value.to_witness())?;
                            sha256_circuit.connect_input(16 + i, hash_wire)?;
                        }
                        
                        for (i, output) in outputs.iter().enumerate() {
                            let output_wire = sha256_circuit.get_output(i)?;
                            self.witness_wire_map.insert(*output, output_wire);
                        }
                        
                        self.merge_subcircuit(sha256_circuit)?;
                    }
                    BlackBoxFuncCall::BigIntAdd { lhs, rhs, output } => {
                        let mut bigint_circuit = self.load_bigint_subcircuit()?;
                        
                        let op_selector_wires = bigint_circuit.add_input_value(2)?;
                        
                        for i in 0..8 {
                            let lhs_wire = self.get_or_create_wire(Witness(*lhs + i as u32))?;
                            bigint_circuit.connect_input(i, lhs_wire)?;
                        }
                        
                        for i in 0..8 {
                            let rhs_wire = self.get_or_create_wire(Witness(*rhs + i as u32))?;
                            bigint_circuit.connect_input(i + 8, rhs_wire)?;
                        }
                        
                        bigint_circuit.connect_input(16, op_selector_wires[0])?;
                        
                        for i in 0..8 {
                            let output_wire = bigint_circuit.get_output(i)?;
                            self.witness_wire_map.insert(Witness(*output + i as u32), output_wire);
                        }
                        
                        self.merge_subcircuit(bigint_circuit)?;
                    }
                    BlackBoxFuncCall::BigIntSub { lhs, rhs, output } => {
                        let mut bigint_circuit = self.load_bigint_subcircuit()?;
                        
                        let op_selector_wires = bigint_circuit.add_input_value(2)?;
                        
                        for i in 0..8 {
                            let lhs_wire = self.get_or_create_wire(Witness(*lhs + i as u32))?;
                            bigint_circuit.connect_input(i, lhs_wire)?;
                        }
                        
                        for i in 0..8 {
                            let rhs_wire = self.get_or_create_wire(Witness(*rhs + i as u32))?;
                            bigint_circuit.connect_input(i + 8, rhs_wire)?;
                        }
                        
                        bigint_circuit.connect_input(16, op_selector_wires[0])?;
                        
                        for i in 0..8 {
                            let output_wire = bigint_circuit.get_output(i)?;
                            self.witness_wire_map.insert(Witness(*output + i as u32), output_wire);
                        }
                        
                        self.merge_subcircuit(bigint_circuit)?;
                    }
                    BlackBoxFuncCall::BigIntMul { lhs, rhs, output } => {
                        let mut bigint_circuit = self.load_bigint_subcircuit()?;
                        
                        let op_selector_wires = bigint_circuit.add_input_value(2)?;
                        
                        for i in 0..8 {
                            let lhs_wire = self.get_or_create_wire(Witness(*lhs + i as u32))?;
                            bigint_circuit.connect_input(i, lhs_wire)?;
                        }
                        
                        for i in 0..8 {
                            let rhs_wire = self.get_or_create_wire(Witness(*rhs + i as u32))?;
                            bigint_circuit.connect_input(i + 8, rhs_wire)?;
                        }
                        
                        bigint_circuit.connect_input(16, op_selector_wires[0])?;
                        
                        for i in 0..8 {
                            let output_wire = bigint_circuit.get_output(i)?;
                            self.witness_wire_map.insert(Witness(*output + i as u32), output_wire);
                        }
                        
                        self.merge_subcircuit(bigint_circuit)?;
                    }
                    BlackBoxFuncCall::BigIntDiv { lhs, rhs, output } => {
                        let mut bigint_circuit = self.load_bigint_subcircuit()?;
                        
                        bigint_circuit.connect_input(0, (*lhs).try_into().unwrap())?;
                        bigint_circuit.connect_input(1, (*rhs).try_into().unwrap())?;
                        
                        let output_wire = bigint_circuit.get_output(0)?;
                        self.witness_wire_map.insert(Witness(*output), output_wire);
                    }
                    _ => return Err("Blackbox function not implemented yet".to_string()),
                }
            }
            _ => return Err("Opcode not implemented yet".to_string()),
        }
        Ok(())
    }

    fn load_blake2s_subcircuit(&self) -> Result<BristolCircuit<F>, String> {
        // TODO: Implement loading of pre-defined Blake2s sub-circuit
        Err("Blake2s sub-circuit loading not implemented yet".to_string())
    }

    fn load_blake3_subcircuit(&self) -> Result<BristolCircuit<F>, String> {
        // TODO: Implement loading of pre-defined Blake3 sub-circuit
        Err("Blake3 sub-circuit loading not implemented yet".to_string())
    }

    fn load_keccak_subcircuit(&self) -> Result<BristolCircuit<F>, String> {
        // TODO: Implement loading of pre-defined Keccak sub-circuit
        Err("Keccak sub-circuit loading not implemented yet".to_string())
    }

    fn load_poseidon2_subcircuit(&self, _len: u32) -> Result<BristolCircuit<F>, String> {
        // TODO: Implement loading of pre-defined Poseidon2 sub-circuit
        Err("Poseidon2 sub-circuit loading not implemented yet".to_string())
    }

    fn load_sha256_subcircuit(&self) -> Result<BristolCircuit<F>, String> {
        let mut circuit = BristolCircuit::new();
        
        let mut input_wires = Vec::new();
        for _ in 0..16 {
            input_wires.push(circuit.add_input_value(32)?);
        }
        
        let mut hash_wires = Vec::new();
        for _ in 0..8 {
            hash_wires.push(circuit.add_input_value(32)?);
        }
        
        let mut output_wires = Vec::new();
        for _ in 0..8 {
            let wire = circuit.add_gate(
                GateType::AND,
                vec![],
                circuit.get_output(0)?,
            )?;
            circuit.add_output_value(32)?;
            output_wires.push(wire);
        }
        
        Ok(circuit)
    }

    fn load_bigint_subcircuit(&self) -> Result<BristolCircuit<F>, String> {
        let mut circuit = BristolCircuit::new();
        
        // Create input wires for two operands (each 256 bits)
        let mut lhs_wires = Vec::new();
        let mut rhs_wires = Vec::new();
        
        // Each operand is represented as 8 32-bit words
        for _ in 0..8 {
            lhs_wires.push(circuit.add_input_value(32)?[0]);
            rhs_wires.push(circuit.add_input_value(32)?[0]);
        }
        
        // Create operation selector input (2 bits)
        let op_selector = circuit.add_input_value(2)?[0];
        
        // Create output wires for result (256 bits)
        let mut result_wires = Vec::new();
        for _ in 0..8 {
            let output_wires = circuit.add_output_value(32)?;
            result_wires.push(output_wires[0]);
        }

        // For each word 
        for i in 0..8 {
            let add_output = circuit.add_output_value(1)?[0];
            circuit.add_gate(
                GateType::AND,
                vec![lhs_wires[i], rhs_wires[i]],
                add_output,
            )?;

            let sub_output = circuit.add_output_value(1)?[0];
            circuit.add_gate(
                GateType::AND,
                vec![lhs_wires[i], rhs_wires[i]],
                sub_output,
            )?;

            let mut mul_results = Vec::new();
            for j in 0..8 {
                if i + j < 8 { 
                    let mul_output = circuit.add_output_value(1)?[0];
                    circuit.add_gate(
                        GateType::AND,
                        vec![lhs_wires[i], rhs_wires[j]],
                        mul_output,
                    )?;
                    mul_results.push(mul_output);
                }
            }

            let const_one = circuit.add_input_value(2)?[0];
            let const_two = circuit.add_input_value(2)?[0];
            
            let op0_output = circuit.add_output_value(1)?[0];
            circuit.add_gate(
                GateType::AND,
                vec![op_selector, const_one],
                op0_output,
            )?;

            let op1_output = circuit.add_output_value(1)?[0];
            circuit.add_gate(
                GateType::AND,
                vec![op_selector, const_two],
                op1_output,
            )?;

            // Select between add and sub results
            let add_gate_output = circuit.add_output_value(1)?[0];
            circuit.add_gate(
                GateType::AND,
                vec![add_output, op0_output],
                add_gate_output,
            )?;

            let sub_gate_output = circuit.add_output_value(1)?[0];
            circuit.add_gate(
                GateType::AND,
                vec![sub_output, op1_output],
                sub_gate_output,
            )?;

            let add_sub_output = circuit.add_output_value(1)?[0];
            circuit.add_gate(
                GateType::AND,
                vec![add_gate_output, sub_gate_output],
                add_sub_output,
            )?;

            // Handle carry/borrow propagation
            if i > 0 {
                let const_carry = circuit.add_input_value(33)?[0];
                let carry_output = circuit.add_output_value(1)?[0];
                circuit.add_gate(
                    GateType::AND,
                    vec![add_sub_output, const_carry],
                    carry_output,
                )?;

                let final_output = circuit.add_output_value(1)?[0];
                circuit.add_gate(
                    GateType::AND,
                    vec![add_sub_output, carry_output],
                    final_output,
                )?;

                result_wires[i] = final_output;
            } else {
                result_wires[i] = add_sub_output;
            }
        }
        
        Ok(circuit)
    }

    fn translate_expression(&mut self, expr: &Expression<F>) -> Result<(), String> {
        // Handle linear combinations
        for (coeff, witness) in &expr.linear_combinations {
            let wire = self.get_or_create_wire(*witness)?;
            if !coeff.is_one() {
                let const_wire = self.circuit.add_input_value(coeff.num_bits())?[0];
                let output_wires = self.circuit.add_output_value(1)?;
                self.circuit.add_gate(
                    GateType::AND,
                    vec![wire, const_wire],
                    output_wires[0],
                )?;
            }
        }

        for (coeff, witness1, witness2) in &expr.mul_terms {
            let wire1 = self.get_or_create_wire(*witness1)?;
            let wire2 = self.get_or_create_wire(*witness2)?;
            
            let output_wires = self.circuit.add_output_value(1)?;
            self.circuit.add_gate(
                GateType::AND,
                vec![wire1, wire2],
                output_wires[0],
            )?;
            
            if !coeff.is_one() {
                let const_wire = self.circuit.add_input_value(coeff.num_bits())?[0];
                let output_wires = self.circuit.add_output_value(1)?;
                self.circuit.add_gate(
                    GateType::AND,
                    vec![output_wires[0], const_wire],
                    output_wires[0],
                )?;
            }
        }

        if !expr.q_c.is_zero() {
            let _ = self.circuit.add_input_value(expr.q_c.num_bits().try_into().unwrap())?;
        }

        Ok(())
    }

    fn merge_subcircuit(&mut self, other: BristolCircuit<F>) -> Result<(), String> {
        self.circuit.merge_subcircuit(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acir_field::FieldElement;
    use acir::circuit::{Circuit, ExpressionWidth, PublicInputs};
    use acir::circuit::opcodes::{BlackBoxFuncCall, FunctionInput};
    use acir::native_types::Witness;
    use std::collections::BTreeSet;

    #[test]
    fn test_simple_and_gate() -> Result<(), String> {
        let mut circuit = BristolCircuit::<FieldElement>::new();
        let input1 = circuit.add_input_value(1)?;
        let input2 = circuit.add_input_value(1)?;
        let output = circuit.add_output_value(1)?;
        circuit.add_gate(
            GateType::AND,
            vec![input1[0], input2[0]],
            output[0],
        )?;
        
        let bristol = circuit.to_bristol_format()?;
        assert!(bristol.contains("AND"));
        Ok(())
    }

    #[test]
    fn test_complex_circuit() -> Result<(), String> {
        let mut circuit = BristolCircuit::<FieldElement>::new();
        let input1 = circuit.add_input_value(1)?;
        let input2 = circuit.add_input_value(1)?;
        let input3 = circuit.add_input_value(1)?;
        
        let xor_output = circuit.add_output_value(1)?;
        circuit.add_gate(
            GateType::XOR,
            vec![input1[0], input2[0]],
            xor_output[0],
        )?;

        let and_output = circuit.add_output_value(1)?;
        circuit.add_gate(
            GateType::AND,
            vec![xor_output[0], input3[0]],
            and_output[0],
        )?;

        let inv_output = circuit.add_output_value(1)?;
        circuit.add_gate(
            GateType::INV,
            vec![and_output[0]],
            inv_output[0],
        )?;
        
        let bristol = circuit.to_bristol_format()?;
        assert!(bristol.contains("XOR"));
        assert!(bristol.contains("AND"));
        assert!(bristol.contains("INV"));
        Ok(())
    }

    #[test]
    fn test_constant_gate() -> Result<(), String> {
        let mut circuit = BristolCircuit::<FieldElement>::new();
        let const_wire = circuit.add_input_value(8)?;
        let output = circuit.add_output_value(8)?;
        
        circuit.add_gate(
            GateType::AND,
            vec![const_wire[0], const_wire[0]],
            output[0],
        )?;
        
        let bristol = circuit.to_bristol_format()?;
        assert!(bristol.contains("AND"));
        Ok(())
    }

    #[test]
    fn test_range_check() -> Result<(), String> {
        let mut circuit = BristolCircuit::<FieldElement>::new();
        let input = circuit.add_input_value(8)?;
        let max = circuit.add_input_value(8)?;
        let output = circuit.add_output_value(8)?;
        
        circuit.add_gate(
            GateType::AND,
            vec![input[0], max[0]],
            output[0],
        )?;
        
        let bristol = circuit.to_bristol_format()?;
        assert!(bristol.contains("AND"));
        Ok(())
    }

    #[test]
    fn test_invalid_circuit() -> Result<(), String> {
        let mut circuit = BristolCircuit::<FieldElement>::new();
        let input1 = circuit.add_input_value(1)?;
        let input2 = circuit.add_input_value(1)?;
        
        let output1 = circuit.add_output_value(1)?;
        assert!(circuit.add_gate(
            GateType::AND,
            vec![input1[0], 999],
            output1[0],
        ).is_err());
        
        let output2 = circuit.add_output_value(1)?;
        circuit.add_gate(
            GateType::AND,
            vec![input1[0], input2[0]],
            output2[0],
        )?;
        
        let output3 = circuit.add_output_value(1)?;
        assert!(circuit.add_gate(
            GateType::AND,
            vec![output2[0], input1[0]],
            output3[0],
        ).is_err());

        Ok(())
    }

    #[test]
    fn test_field_element_operations() -> Result<(), String> {
        let mut circuit = BristolCircuit::<FieldElement>::new();
        let const_wire = circuit.add_input_value(8)?;
        let output = circuit.add_output_value(8)?;
        
        circuit.add_gate(
            GateType::AND,
            vec![const_wire[0], const_wire[0]],
            output[0],
        )?;
        
        let bristol = circuit.to_bristol_format()?;
        assert!(bristol.contains("AND"));
        Ok(())
    }

    #[test]
    fn test_bigint_subcircuit() -> Result<(), String> {
        let mut translator = BristolCircuitTranslator::<FieldElement>::new();
        let bigint_circuit = translator.load_bigint_subcircuit()?;
        
        let bristol = bigint_circuit.to_bristol_format()?;
        
        assert!(bristol.contains("AND"), "Sub-circuit should contain AND gates");
        assert!(bristol.contains("AND"), "Sub-circuit should contain AND gates for operation selection");
        assert!(bristol.contains("XOR"), "Sub-circuit should contain XOR gates for operation selection");
        
        let lines: Vec<&str> = bristol.lines().collect();
        let header: Vec<&str> = lines[0].split_whitespace().collect();
        let num_gates = header[0].parse::<usize>().unwrap();
        let num_wires = header[1].parse::<usize>().unwrap();
        

        assert!(num_gates > 0, "Sub-circuit should have gates");
        assert!(num_wires >= 25, "Sub-circuit should have at least 25 wires (17 inputs + 8 outputs)");

        for (i, usage) in bigint_circuit.wire_usage.iter().enumerate() {
            if *usage == FieldElement::zero() && bigint_circuit.wire_types[i] != WireType::Output {
                return Err(format!("Wire {} is unused", i));
            }
        }

        Ok(())
    }
}

fn main() {
    let circuit = Circuit::<FieldElement> {
        current_witness_index: 27, // 3 witnesses per operation * 3 operations
        opcodes: vec![
            // BigInt addition: witness_1 + witness_2 = witness_3
            Opcode::BlackBoxFuncCall(BlackBoxFuncCall::BigIntAdd {
                lhs: 1,
                rhs: 2,
                output: 3,
            }),
            // BigInt subtraction: witness_4 - witness_5 = witness_6
            Opcode::BlackBoxFuncCall(BlackBoxFuncCall::BigIntSub {
                lhs: 4,
                rhs: 5,
                output: 6,
            }),
            // BigInt multiplication: witness_7 * witness_8 = witness_9
            Opcode::BlackBoxFuncCall(BlackBoxFuncCall::BigIntMul {
                lhs: 7,
                rhs: 8,
                output: 9,
            }),
        ],
        expression_width: ExpressionWidth::Unbounded,
        private_parameters: BTreeSet::new(),
        public_parameters: PublicInputs(BTreeSet::from_iter(vec![
            Witness(1), Witness(2), // Addition inputs
            Witness(4), Witness(5), // Subtraction inputs
            Witness(7), Witness(8), // Multiplication inputs
        ])),
        return_values: PublicInputs(BTreeSet::from_iter(vec![
            Witness(3), // Addition result
            Witness(6), // Subtraction result
            Witness(9), // Multiplication result
        ])),
        assert_messages: Vec::new(),
    };

    let mut translator = BristolCircuitTranslator::<FieldElement>::new();
    if let Err(e) = translator.translate_circuit(&circuit) {
        eprintln!("Error translating circuit: {}", e);
        return;
    }

    let bristol = match translator.circuit.to_bristol_format() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error generating Bristol format: {}", e);
            return;
        }
    };
    
    println!("\nInput Circuit:");
    println!("{}", circuit);
    println!("\nBristol Format Output:");
    println!("{}", bristol);
    
    let lines: Vec<&str> = bristol.lines().collect();
    let header: Vec<&str> = lines[0].split_whitespace().collect();
    let num_gates = header[0].parse::<usize>().unwrap();
    let num_wires = header[1].parse::<usize>().unwrap();
    println!("\nCircuit Statistics:");
    println!("Number of gates: {}", num_gates);
    println!("Number of wires: {}", num_wires);
} 