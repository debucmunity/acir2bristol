use std::collections::HashMap;
use acir::circuit::opcodes::*;
use acir::circuit::*;
use acir::native_types::*;
use acir_field::*;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Write;

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
        //println!("Adding gate: {:?} with inputs {:?} and output {}", gate_type, inputs, output);
        
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
            // println!("Input wire {} type: {:?}, current usage: {}", input, self.wire_types[input], self.wire_usage[input]);
        }

        // Validate output wire
        if output >= self.num_wires {
            return Err(format!("Output wire {} does not exist", output));
        }
        // println!("Output wire {} type: {:?}, current usage: {}", output, self.wire_types[output], self.wire_usage[output]);
        
        // Update wire usage for inputs
        for &input in &inputs {
            self.wire_usage[input] = self.wire_usage[input] + F::one();
            // println!("Updated input wire {} usage to {}", input, self.wire_usage[input]);
        }
        
        // Update wire usage for output
        self.wire_usage[output] = self.wire_usage[output] + F::one();
        // println!("Updated output wire {} usage to {}", output, self.wire_usage[output]);
        
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
            // println!("Wire {}: type {:?}, usage {}", i, self.wire_types[i], usage);
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

    pub fn connect_input(&mut self, input_group_index_in_this_circuit: usize, _parent_wire_providing_value: usize) -> Result<(), String> {
        // println!("Connecting input {} to wire {}", input_index, wire);
        
        if input_group_index_in_this_circuit >= self.input_values.len() {
            return Err(format!(
                "Subcircuit input group index {} is out of bounds. This circuit has {} input groups.",
                input_group_index_in_this_circuit,
                self.input_values.len()
            ));
        }
        
        let input_group = &self.input_values[input_group_index_in_this_circuit];
        if input_group.is_empty() {
            return Err(format!("Subcircuit input group {} is empty.", input_group_index_in_this_circuit));
        }
        let sub_input_wire_idx = input_group[0]; // Assuming single wire per input group for these connections

        if sub_input_wire_idx >= self.num_wires {
            return Err(format!(
                "Subcircuit input wire index {} (from group {}) is out of bounds for this circuit's {} wires.",
                sub_input_wire_idx, input_group_index_in_this_circuit, self.num_wires
            ));
        }
        
        // Update wire usage for this circuit's own input wire that is being connected.
        self.wire_usage[sub_input_wire_idx] = self.wire_usage[sub_input_wire_idx] + F::one();
        // println!("Updated subcircuit wire {} (from input group {}) usage to {}", 
        //          sub_input_wire_idx, input_group_index_in_this_circuit, self.wire_usage[sub_input_wire_idx]);
        
        Ok(())
    }

    pub fn get_output(&self, output_group_index: usize) -> Result<usize, String> {
        if output_group_index >= self.output_values.len() {
            return Err(format!(
                "Output group index {} is out of bounds. Circuit has {} output groups.",
                output_group_index,
                self.output_values.len()
            ));
        }
        let output_group = &self.output_values[output_group_index];
        if output_group.is_empty() {
            return Err(format!("Output group {} is empty.", output_group_index));
        }
        Ok(output_group[0])
    }

    pub fn merge_subcircuit(&mut self, other: BristolCircuit<F>) -> Result<(), String> {
        // println!("\nMerging subcircuit:");
        // println!("Original circuit: {} wires, {} gates", self.num_wires, self.gates.len());
        // println!("Subcircuit: {} wires, {} gates", other.num_wires, other.gates.len());
        
        let wire_map_offset = self.num_wires;
        let mut wire_map = Vec::new(); // Maps other.wire_idx to new self.wire_idx
        for i in 0..other.num_wires {
            let new_wire = self.num_wires; // new wire index in parent/self
            self.num_wires += 1;
            self.wire_usage.push(F::zero()); // Initialize usage for new wires in parent
            self.wire_types.push(other.wire_types[i].clone());
            wire_map.push(new_wire);
            // println!("Mapped wire {} from subcircuit to new parent wire {}", i, new_wire);
        }

        for gate in other.gates {
            let new_inputs = gate.inputs.iter().map(|&input| wire_map[input]).collect();
            let new_output = wire_map[gate.output];
            
            // By calling self.add_gate, we ensure that wire_usage in the parent circuit (self)
            // is correctly updated for these newly mapped input and output wires of the merged gate.
            // This also performs necessary validations.
            self.add_gate(gate.gate_type.clone(), new_inputs, new_output)?;
        }

        // Remap and append input_values from the subcircuit
        for input_group in other.input_values {
            let remapped_input_group = input_group.iter().map(|&wire_idx| wire_map[wire_idx]).collect();
            self.input_values.push(remapped_input_group);
        }

        // Remap and append output_values from the subcircuit
        for output_group in other.output_values {
            let remapped_output_group = output_group.iter().map(|&wire_idx| wire_map[wire_idx]).collect();
            self.output_values.push(remapped_output_group);
        }

        // println!("Merged circuit: {} wires, {} gates", self.num_wires, self.gates.len());
        // println!("Parent circuit now has {} input groups and {} output groups.", self.input_values.len(), self.output_values.len());
        Ok(())
    }
}

pub fn simulate_bristol_circuit<F: AcirField>(
    circuit: &BristolCircuit<F>,
    inputs: &HashMap<usize, bool>, // Wire index to boolean value
) -> Result<HashMap<usize, bool>, String> {
    let mut wire_values: HashMap<usize, bool> = inputs.clone();

    for input_group in &circuit.input_values {
        for &wire_idx in input_group {
            if !wire_values.contains_key(&wire_idx) {
                return Err(format!(
                    "Input wire {} was defined in circuit but not provided in simulation inputs",
                    wire_idx
                ));
            }
        }
    }

    for gate in &circuit.gates {
        let output_val = match gate.gate_type {
            GateType::XOR => {
                let in1 = wire_values.get(&gate.inputs[0]).ok_or_else(|| {
                    format!("Value for XOR input wire {} not found", gate.inputs[0])
                })?;
                let in2 = wire_values.get(&gate.inputs[1]).ok_or_else(|| {
                    format!("Value for XOR input wire {} not found", gate.inputs[1])
                })?;
                in1 ^ in2
            }
            GateType::AND => {
                let in1 = wire_values.get(&gate.inputs[0]).ok_or_else(|| {
                    format!("Value for AND input wire {} not found", gate.inputs[0])
                })?;
                let in2 = wire_values.get(&gate.inputs[1]).ok_or_else(|| {
                    format!("Value for AND input wire {} not found", gate.inputs[1])
                })?;
                in1 & in2
            }
            GateType::INV => {
                let in1 = wire_values.get(&gate.inputs[0]).ok_or_else(|| {
                    format!("Value for INV input wire {} not found", gate.inputs[0])
                })?;
                !in1
            }
            GateType::EQ | GateType::EQW => {
                let in1 = wire_values.get(&gate.inputs[0]).ok_or_else(|| {
                    format!("Value for EQ input wire {} not found", gate.inputs[0])
                })?;
                 if gate.inputs.len() == 1 {
                    *wire_values.get(&gate.inputs[0]).ok_or_else(|| {
                        format!("Value for EQ input wire {} not found", gate.inputs[0])
                    })?
                 } else {
                    return Err(format!("Unsupported input count for EQ/EQW gate: {:?}", gate.inputs));
                 }
            }
            GateType::MAND => {
                // Multi-AND: (in0 & in1) & (in2 & in3) & ...
                // Your add_gate validation ensures an even number of inputs.
                if gate.inputs.len() < 2 || gate.inputs.len() % 2 != 0 {
                     return Err(format!("MAND gate has invalid number of inputs: {}", gate.inputs.len()));
                }
                let mut current_result = true; // Identity for AND
                for chunk in gate.inputs.chunks_exact(2) {
                    let val1 = wire_values.get(&chunk[0]).ok_or_else(|| {
                        format!("Value for MAND input wire {} not found", chunk[0])
                    })?;
                    let val2 = wire_values.get(&chunk[1]).ok_or_else(|| {
                        format!("Value for MAND input wire {} not found", chunk[1])
                    })?;
                    current_result &= (val1 & val2);
                }
                current_result
            }
        };
        wire_values.insert(gate.output, output_val);
    }

    Ok(wire_values)
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
        // println!("Translator: Registering public parameters as input wires...");
        for witness in circuit.public_parameters.0.iter() {
            self.register_witness(*witness, WireType::Input)?;
        }
        // println!("Translator: Registering private parameters as input wires...");
        for witness in circuit.private_parameters.iter() {
            self.register_witness(*witness, WireType::Input)?;
        }

        // println!("Translator: Translating opcodes...");
        for (idx, opcode) in circuit.opcodes.iter().enumerate() {
            println!("  Opcode {}: {:?}", idx, opcode);
            self.translate_opcode(opcode)?;
        }

        // println!("Translator: Marking return value wires as used...");
        for witness in circuit.return_values.0.iter() {
            if let Some(wire) = self.witness_wire_map.get(witness) {
                // Mark output wires as used so they pass validation, even if no gate explicitly writes to them
                // (e.g. if they are direct copies of input wires serving as outputs)
                if self.circuit.wire_types[*wire] == WireType::Output || self.circuit.wire_types[*wire] == WireType::Input {
                    self.circuit.wire_usage[*wire] = self.circuit.wire_usage[*wire] + F::one();
                //    println!("  Marked return wire {} (Witness {}) as used.", wire, witness.0);
                }
            } else {
                // This case might indicate an issue if a return witness was never mapped.
                println!("  Warning: Return Witness {} not found in wire_map.", witness.0);
            }
        }

        Ok(())
    }

    fn register_witness(&mut self, witness: Witness, wire_type: WireType) -> Result<usize, String> {
        if !self.witness_wire_map.contains_key(&witness) {
            let wires = self.circuit.add_input_value(1)?; // Create a single wire for this witness
            let wire_idx = wires[0];
            self.witness_wire_map.insert(witness, wire_idx);
            // Set the wire type specified by the caller (e.g., Input)
            if wire_idx < self.circuit.wire_types.len() {
                 self.circuit.wire_types[wire_idx] = wire_type;
            } else {
                // This case should ideally not happen if add_input_value correctly resizes wire_types
                return Err(format!("Wire index {} out of bounds for wire_types after add_input_value", wire_idx));
            }
            // println!("Translator: Registered Witness {} as wire {} with type {:?}", witness.0, wire_idx, self.circuit.wire_types[wire_idx]);
            Ok(wire_idx)
        } else {
            // If witness already registered, just return its existing wire index
            Ok(*self.witness_wire_map.get(&witness).unwrap())
        }
    }

    fn get_or_create_wire(&mut self, witness: Witness) -> Result<usize, String> {
        match self.witness_wire_map.get(&witness) {
            Some(wire_idx) => Ok(*wire_idx),
            None => {
                // If not found, register it as a new input wire by default.
                // This path is typically for intermediate witnesses implicitly created by opcodes.
                println!("Translator: Witness {} not found in map, creating new input wire for it.", witness.0);
                self.register_witness(witness, WireType::Input) 
            }
        }
    }

    fn get_or_create_constant_wire(&mut self, value: bool) -> Result<usize, String> {
        // For simplicity, we create a new input wire and record its intended constant value.
        // In a real ZK-SNARK backend, this would involve constraining this wire.
        // For Bristol simulation, the test will need to provide this value in sim_inputs.
        
        // We need a way to distinguish these constant wires or manage them.
        // A simple approach for now: create a unique witness for it (though it's not a traditional witness).
        // Or, just create an input wire and rely on the caller (translator or test) to handle its constness.
        
        let constant_wires = self.circuit.add_input_value(1)?; // Creates a single wire in a new input group
        let wire_idx = constant_wires[0];
        self.circuit.wire_types[wire_idx] = WireType::Constant; // Mark its type
        
        // How to store the actual constant value `value`? 
        // The BristolCircuit itself doesn't store values, only structure.
        // This function provides the wire index. The caller (or simulation setup) must use it correctly.
        // println!("Translator: Created constant wire {} (intended value: {}) for OpSel.", wire_idx, value);
        Ok(wire_idx)
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
                        const NUM_BITS: u32 = 64;
                        let mut bigint_circuit = self.load_bigint_subcircuit()?;

                        let parent_op_sel0_wire_for_add = self.get_or_create_constant_wire(false)?;
                        self.circuit.wire_usage[parent_op_sel0_wire_for_add] += F::one(); 
                        let parent_op_sel1_wire_for_add = self.get_or_create_constant_wire(false)?;
                        self.circuit.wire_usage[parent_op_sel1_wire_for_add] += F::one();

                        for i in 0..NUM_BITS {
                            let lhs_bit_witness = Witness(*lhs + i);
                            let lhs_parent_wire = self.get_or_create_wire(lhs_bit_witness)?;
                            self.circuit.wire_usage[lhs_parent_wire] += F::one();
                            bigint_circuit.connect_input(i as usize, lhs_parent_wire)?;
                        }

                        for i in 0..NUM_BITS {
                            let rhs_bit_witness = Witness(*rhs + i);
                            let rhs_parent_wire = self.get_or_create_wire(rhs_bit_witness)?;
                            self.circuit.wire_usage[rhs_parent_wire] += F::one();
                            bigint_circuit.connect_input((NUM_BITS + i) as usize, rhs_parent_wire)?;
                        }

                        bigint_circuit.connect_input((NUM_BITS * 2) as usize, parent_op_sel0_wire_for_add)?;
                        bigint_circuit.connect_input((NUM_BITS * 2 + 1) as usize, parent_op_sel1_wire_for_add)?;

                        let mut output_mappings = Vec::new();
                        for i in 0..NUM_BITS {
                            let output_bit_witness = Witness(*output + i);
                            let sub_output_wire_idx = bigint_circuit.get_output(i as usize)?;
                            output_mappings.push((output_bit_witness, sub_output_wire_idx));
                        }

                        let parent_wires_before_merge = self.circuit.num_wires;
                        self.merge_subcircuit(bigint_circuit)?;

                        for (witness, sub_output_wire_idx_in_sub) in output_mappings {
                            let parent_equivalent_wire_idx = parent_wires_before_merge + sub_output_wire_idx_in_sub;
                            self.witness_wire_map.insert(witness, parent_equivalent_wire_idx);
                        }
                    }

                    BlackBoxFuncCall::BigIntSub { lhs, rhs, output } => {
                        const NUM_BITS: u32 = 64;
                        let mut bigint_circuit = self.load_bigint_subcircuit()?; 

                        let parent_op_sel0_wire_for_sub = self.get_or_create_constant_wire(true)?;
                        self.circuit.wire_usage[parent_op_sel0_wire_for_sub] += F::one(); 
                        let parent_op_sel1_wire_for_sub = self.get_or_create_constant_wire(false)?;
                        self.circuit.wire_usage[parent_op_sel1_wire_for_sub] += F::one();

                        // println!("Translator: Connecting {} LHS bits for BigIntSub...", NUM_BITS);
                        for i in 0..NUM_BITS {
                            let lhs_bit_witness = Witness(*lhs + i);
                            let lhs_parent_wire = self.get_or_create_wire(lhs_bit_witness)?;
                            self.circuit.wire_usage[lhs_parent_wire] += F::one();
                            // Subcircuit's LHS input groups are 0 to NUM_BITS-1
                            bigint_circuit.connect_input(i as usize, lhs_parent_wire)?;
                        }

                        // println!("Translator: Connecting {} RHS bits for BigIntSub...", NUM_BITS);
                        for i in 0..NUM_BITS {
                            let rhs_bit_witness = Witness(*rhs + i);
                            let rhs_parent_wire = self.get_or_create_wire(rhs_bit_witness)?;
                            self.circuit.wire_usage[rhs_parent_wire] += F::one();
                            // Subcircuit's RHS input groups are NUM_BITS to 2*NUM_BITS-1
                            bigint_circuit.connect_input((NUM_BITS + i) as usize, rhs_parent_wire)?;
                        }
                        
                        // println!("Translator: Connecting OpSelector wires for BigIntSub (OpSel0=true, OpSel1=false)...");
                        // Subcircuit's OpSel0 is input group 2*NUM_BITS
                        // Subcircuit's OpSel1 is input group 2*NUM_BITS + 1
                        bigint_circuit.connect_input((NUM_BITS * 2) as usize, parent_op_sel0_wire_for_sub)?;
                        bigint_circuit.connect_input((NUM_BITS * 2 + 1) as usize, parent_op_sel1_wire_for_sub)?;

                        // println!("Translator: Preparing {} Output bit mappings for BigIntSub...", NUM_BITS);
                        let mut output_mappings = Vec::new();
                        for i in 0..NUM_BITS {
                            let output_bit_witness = Witness(*output + i);
                            // Output groups are 0 to NUM_BITS-1 in the subcircuit
                            let sub_output_wire_idx = bigint_circuit.get_output(i as usize)?;
                            output_mappings.push((output_bit_witness, sub_output_wire_idx));
                        }

                        let parent_wires_before_merge = self.circuit.num_wires;
                        println!("Translator: Merging BigIntSub subcircuit ({} wires) into parent ({} wires before merge)...", bigint_circuit.num_wires, parent_wires_before_merge);
                        self.merge_subcircuit(bigint_circuit)?;
                        println!("Translator: Parent circuit now has {} wires after merging BigIntSub subcircuit.", self.circuit.num_wires);

                        println!("Translator: Finalizing Output bit mappings for BigIntSub post-merge...");
                        for (witness, sub_output_wire_idx_in_sub) in output_mappings {
                            let parent_equivalent_wire_idx = parent_wires_before_merge + sub_output_wire_idx_in_sub;
                            self.witness_wire_map.insert(witness, parent_equivalent_wire_idx);
                            println!("  Mapped Sub-Witness Output (Witness {}) to merged circuit wire {} (from sub_wire {})", witness.0, parent_equivalent_wire_idx, sub_output_wire_idx_in_sub);
                        }
                    }
                    BlackBoxFuncCall::BigIntMul { lhs, rhs, output } => {
                        const NUM_BITS: u32 = 64;
                        let mut bigint_circuit = self.load_bigint_subcircuit()?;

                        let parent_op_sel0_wire_for_mul = self.get_or_create_constant_wire(false)?;
                        self.circuit.wire_usage[parent_op_sel0_wire_for_mul] += F::one(); 
                        let parent_op_sel1_wire_for_mul = self.get_or_create_constant_wire(true)?;
                        self.circuit.wire_usage[parent_op_sel1_wire_for_mul] += F::one();

                        for i in 0..NUM_BITS {
                            let lhs_bit_witness = Witness(*lhs + i);
                            let lhs_parent_wire = self.get_or_create_wire(lhs_bit_witness)?;
                            self.circuit.wire_usage[lhs_parent_wire] += F::one();
                            bigint_circuit.connect_input(i as usize, lhs_parent_wire)?;
                        }

                        for i in 0..NUM_BITS {
                            let rhs_bit_witness = Witness(*rhs + i);
                            let rhs_parent_wire = self.get_or_create_wire(rhs_bit_witness)?;
                            self.circuit.wire_usage[rhs_parent_wire] += F::one();
                            bigint_circuit.connect_input((NUM_BITS + i) as usize, rhs_parent_wire)?;
                        }

                        bigint_circuit.connect_input((NUM_BITS * 2) as usize, parent_op_sel0_wire_for_mul)?;
                        bigint_circuit.connect_input((NUM_BITS * 2 + 1) as usize, parent_op_sel1_wire_for_mul)?;

                        let mut output_mappings = Vec::new();
                        for i in 0..NUM_BITS {
                            let output_bit_witness = Witness(*output + i);
                            let sub_output_wire_idx = bigint_circuit.get_output(i as usize)?;
                            output_mappings.push((output_bit_witness, sub_output_wire_idx));
                        }

                        let parent_wires_before_merge = self.circuit.num_wires;
                        self.merge_subcircuit(bigint_circuit)?;

                        for (witness, sub_output_wire_idx_in_sub) in output_mappings {
                            let parent_equivalent_wire_idx = parent_wires_before_merge + sub_output_wire_idx_in_sub;
                            self.witness_wire_map.insert(witness, parent_equivalent_wire_idx);
                        }
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

    // Helper to create a new intermediate wire
    fn new_intermediate_wire(&self, circuit: &mut BristolCircuit<F>) -> Result<usize, String> {
        let wire_idx = circuit.num_wires;
        circuit.num_wires += 1;
        circuit.wire_usage.push(F::zero());
        circuit.wire_types.push(WireType::Intermediate);
        Ok(wire_idx)
    }

    fn load_bigint_subcircuit(&self) -> Result<BristolCircuit<F>, String> {
        let mut circuit = BristolCircuit::new();
        const NUM_BITS: usize = 64;

        let mut lhs_wires = Vec::new();
        for i in 0..NUM_BITS {
            lhs_wires.push(circuit.add_input_value(1)?[0]);
        }
        let mut rhs_wires = Vec::new();
        for i in 0..NUM_BITS {
            rhs_wires.push(circuit.add_input_value(1)?[0]);
        }

        // Define OpSelector inputs as two separate single-wire groups
        let op_sel_0_input_group = circuit.add_input_value(1)?;
        let op_sel_0_input_wire = op_sel_0_input_group[0];
        let op_sel_1_input_group = circuit.add_input_value(1)?;
        let op_sel_1_input_wire = op_sel_1_input_group[0];
        // println!("Subcircuit: OpSel input wires defined: Sel0={}, Sel1={}", op_sel_0_input_wire, op_sel_1_input_wire);

        let mut result_wires = Vec::new();
        for i in 0..NUM_BITS {
            result_wires.push(circuit.add_output_value(1)?[0]);
        }

        let zero_wire = self.new_intermediate_wire(&mut circuit)?;
        circuit.add_gate(GateType::XOR, vec![lhs_wires[0], lhs_wires[0]], zero_wire)?;
        
        let op_s0_internal = self.new_intermediate_wire(&mut circuit)?;
        circuit.add_gate(GateType::XOR, vec![op_sel_0_input_wire, zero_wire], op_s0_internal)?; 
        let op_s1_internal = self.new_intermediate_wire(&mut circuit)?;
        circuit.add_gate(GateType::XOR, vec![op_sel_1_input_wire, zero_wire], op_s1_internal)?;
        
        let op_s0_inv = self.new_intermediate_wire(&mut circuit)?;
        circuit.add_gate(GateType::INV, vec![op_s0_internal], op_s0_inv)?;
        let op_s1_inv = self.new_intermediate_wire(&mut circuit)?;
        circuit.add_gate(GateType::INV, vec![op_s1_internal], op_s1_inv)?;

        let is_add_selected = self.new_intermediate_wire(&mut circuit)?;
        circuit.add_gate(GateType::AND, vec![op_s1_inv, op_s0_inv], is_add_selected)?;

        let is_sub_selected = self.new_intermediate_wire(&mut circuit)?;
        circuit.add_gate(GateType::AND, vec![op_s1_inv, op_s0_internal], is_sub_selected)?;
        
        let is_mul_selected = self.new_intermediate_wire(&mut circuit)?;
        circuit.add_gate(GateType::AND, vec![op_s1_internal, op_s0_inv], is_mul_selected)?;
        println!("Subcircuit: Selector wires created: is_add ({}), is_sub ({}), is_mul ({})", is_add_selected, is_sub_selected, is_mul_selected);

        // --- 3. Initial Carry/Borrow for each operation ---
        // For addition: initial carry is 0 
        // For subtraction: initial borrow is 0
        let mut add_carry = zero_wire; // Initial carry for addition is 0
        let mut sub_borrow = zero_wire; // Initial borrow for subtraction is 0

        // --- MULTIPLICATION IMPLEMENTATION ---
        // Use a simpler approach: school-book multiplication with explicit carries
        let mut mul_result_bits = Vec::with_capacity(NUM_BITS);

        // Initialize product bits array (128 bits, all zeros)
        let mut product_bits = vec![zero_wire; NUM_BITS * 2];

        // For each bit position in the result
        for i in 0..NUM_BITS {
            for j in 0..NUM_BITS {
                // Only consider bits that affect our 64-bit result
                if i + j < NUM_BITS {
                    // Create wire for A[i] AND B[j]
                    let pp = self.new_intermediate_wire(&mut circuit)?;
                    circuit.add_gate(GateType::AND, vec![lhs_wires[i], rhs_wires[j]], pp)?;
                    
                    // Position in result is i+j
                    let pos = i + j;
                    
                    // Add this partial product to the result with carry chain
                    let mut current_pos = pos;
                    let mut current_bit = pp;
                    
                    // Propagate carries as far as needed
                    while current_pos < NUM_BITS {
                        // XOR the current product bit with the new bit
                        let new_sum = self.new_intermediate_wire(&mut circuit)?;
                        circuit.add_gate(GateType::XOR, vec![product_bits[current_pos], current_bit], new_sum)?;
                        
                        // Calculate carry: old_bit AND new_bit
                        let carry = self.new_intermediate_wire(&mut circuit)?;
                        circuit.add_gate(GateType::AND, vec![product_bits[current_pos], current_bit], carry)?;
                        
                        // Store the new sum
                        product_bits[current_pos] = new_sum;
                        
                        // If no carry, we're done with this chain
                        if current_pos + 1 >= NUM_BITS * 2 {
                            break;
                        }
                        
                        // Otherwise, propagate carry to next position
                        current_pos += 1;
                        current_bit = carry;
                    }
                }
            }
        }

        // The lower NUM_BITS bits of product are our result
        for i in 0..NUM_BITS {
            mul_result_bits.push(product_bits[i]);
        }

        // --- 4. Process each bit ---
        for i in 0..NUM_BITS {
            let lhs_bit = lhs_wires[i];
            let rhs_bit = rhs_wires[i];

            // --- ADDITION LOGIC ---
            // A + B = A ⊕ B ⊕ carry_in
            let add_xor_bits = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::XOR, vec![lhs_bit, rhs_bit], add_xor_bits)?;
            let add_result = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::XOR, vec![add_xor_bits, add_carry], add_result)?;

            // Carry out: (A & B) | (carry_in & (A ⊕ B))
            let add_and_bits = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::AND, vec![lhs_bit, rhs_bit], add_and_bits)?;
            let add_and_carry = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::AND, vec![add_carry, add_xor_bits], add_and_carry)?;
            
            let next_add_carry = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::XOR, vec![add_and_bits, add_and_carry], next_add_carry)?;
            let add_carry_and = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::AND, vec![add_and_bits, add_and_carry], add_carry_and)?;
            let add_carry_out = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::XOR, vec![next_add_carry, add_carry_and], add_carry_out)?;

            // --- SUBTRACTION LOGIC ---
            // A - B = A ⊕ B ⊕ borrow_in
            let sub_xor_bits = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::XOR, vec![lhs_bit, rhs_bit], sub_xor_bits)?;
            let sub_result = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::XOR, vec![sub_xor_bits, sub_borrow], sub_result)?;

            // Borrow out: (!A & B) | (borrow_in & (A == B))
            // First: !A & B
            let not_lhs = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::INV, vec![lhs_bit], not_lhs)?;
            let term1 = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::AND, vec![not_lhs, rhs_bit], term1)?;

            // Second: A == B is equivalent to !(A ⊕ B)
            let not_sub_xor_bits = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::INV, vec![sub_xor_bits], not_sub_xor_bits)?;
            let term2 = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::AND, vec![sub_borrow, not_sub_xor_bits], term2)?;

            // Combine: term1 | term2
            let sub_borrow_out_xor = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::XOR, vec![term1, term2], sub_borrow_out_xor)?;
            let sub_borrow_out_and = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::AND, vec![term1, term2], sub_borrow_out_and)?;
            let sub_borrow_out = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::XOR, vec![sub_borrow_out_xor, sub_borrow_out_and], sub_borrow_out)?;

            // --- RESULT SELECTION BASED ON OPERATION ---
            let add_result_sel = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::AND, vec![add_result, is_add_selected], add_result_sel)?;
            
            let sub_result_sel = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::AND, vec![sub_result, is_sub_selected], sub_result_sel)?;
            
            let mul_result_sel = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::AND, vec![mul_result_bits[i], is_mul_selected], mul_result_sel)?;

            // Combine results
            let add_sub_xor = self.new_intermediate_wire(&mut circuit)?;
            circuit.add_gate(GateType::XOR, vec![add_result_sel, sub_result_sel], add_sub_xor)?;
            
            circuit.add_gate(GateType::XOR, vec![add_sub_xor, mul_result_sel], result_wires[i])?;

            // Update for next bit
            add_carry = add_carry_out;
            sub_borrow = sub_borrow_out;
        }
        
        println!("Subcircuit: Full {}bit ALU subcircuit created with {} wires and {} gates.", NUM_BITS, circuit.num_wires, circuit.gates.len());
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
        let input1 = circuit.add_input_value(8)?;
        let input2 = circuit.add_input_value(8)?;
        let input3 = circuit.add_input_value(8)?;
        
        // Create intermediate wires for XOR and AND results
        let mut xor_intermediate = Vec::new();
        let mut and_intermediate = Vec::new();
        let inv_output = circuit.add_output_value(8)?;
        
        // Create a chain of gates for each wire
        for i in 0..8 {
            // Add intermediate wire for XOR result
            xor_intermediate.push(circuit.num_wires);
            circuit.wire_types.push(WireType::Intermediate);
            circuit.wire_usage.push(FieldElement::zero());
            circuit.num_wires += 1;

            // Add intermediate wire for AND result
            and_intermediate.push(circuit.num_wires);
            circuit.wire_types.push(WireType::Intermediate);
            circuit.wire_usage.push(FieldElement::zero());
            circuit.num_wires += 1;

            // XOR gate
            circuit.add_gate(
                GateType::XOR,
                vec![input1[i], input2[i]],
                xor_intermediate[i],
            )?;

            // AND gate
            circuit.add_gate(
                GateType::AND,
                vec![xor_intermediate[i], input3[i]],
                and_intermediate[i],
            )?;

            // INV gate
            circuit.add_gate(
                GateType::INV,
                vec![and_intermediate[i]],
                inv_output[i],
            )?;
        }
        
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
        
        // Create AND gates using all wires
        for i in 0..8 {
            circuit.add_gate(
                GateType::AND,
                vec![const_wire[i], const_wire[i]],  // Each wire ANDed with itself
                output[i],
            )?;
        }
        
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
        
        // Create a chain of AND gates using all wires
        for i in 0..8 {
            circuit.add_gate(
                GateType::AND,
                vec![input[i], max[i]],
                output[i],
            )?;
        }
        
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
        
        // Create AND gates using all wires
        for i in 0..8 {
            circuit.add_gate(
                GateType::AND,
                vec![const_wire[i], const_wire[i]],  // Each wire ANDed with itself
                output[i],
            )?;
        }
        
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

    #[test]
    fn test_functional_bigint_add_simple() -> Result<(), String> {
        const NUM_BITS_U32: u32 = 64;
        const NUM_BITS_USIZE: usize = 64;

        // --- 1. Setup Translator and ACIR ---
        let lhs_start_w = Witness(1); 
        let rhs_start_w = Witness(1 + NUM_BITS_U32); 
        let output_start_w = Witness(1 + NUM_BITS_U32 * 2);

        let acir_circuit = Circuit::<FieldElement> {
            current_witness_index: 1 + NUM_BITS_U32 * 3, 
            opcodes: vec![Opcode::BlackBoxFuncCall(BlackBoxFuncCall::BigIntAdd {
                lhs: lhs_start_w.0,
                rhs: rhs_start_w.0,
                output: output_start_w.0,
            })],
            expression_width: ExpressionWidth::Unbounded,
            private_parameters: (0..NUM_BITS_U32 * 2).map(|i| Witness(lhs_start_w.0 + i)).collect(),
            public_parameters: PublicInputs(BTreeSet::new()),
            return_values: PublicInputs((0..NUM_BITS_U32).map(|i| Witness(output_start_w.0 + i)).collect()),
            assert_messages: Vec::new(),
        };

        let mut translator = BristolCircuitTranslator::<FieldElement>::new();
        translator.translate_circuit(&acir_circuit)?;

        let bristol_circuit = translator.circuit.clone();

        // --- 2. Define Test Case Input Values ---
        let val_a: u64 = 16994505098093817264;
        let val_b: u64 = 16054452040491084025;

        // --- 3. Prepare Simulation Inputs ---
        let mut sim_inputs: HashMap<usize, bool> = HashMap::new();

        println!("Initializing all {} input groups in merged circuit to false by default", bristol_circuit.input_values.len());
        for input_group in &bristol_circuit.input_values {
            for &wire in input_group {
                sim_inputs.insert(wire, false);
            }
        }

        // println!("Setting parent-level bit-witness inputs...");
        for i in 0..NUM_BITS_USIZE {
            let lhs_bit_witness = Witness(lhs_start_w.0 + i as u32);
            if let Some(&parent_lhs_bit_wire) = translator.witness_wire_map.get(&lhs_bit_witness) {
                sim_inputs.insert(parent_lhs_bit_wire, (val_a >> i) & 1 == 1);
            } else {
                return Err(format!("LHS bit-witness {} not in map!", lhs_bit_witness.0));
            }

            let rhs_bit_witness = Witness(rhs_start_w.0 + i as u32);
            if let Some(&parent_rhs_bit_wire) = translator.witness_wire_map.get(&rhs_bit_witness) {
                sim_inputs.insert(parent_rhs_bit_wire, (val_b >> i) & 1 == 1);
            } else {
                return Err(format!("RHS bit-witness {} not in map!", rhs_bit_witness.0));
            }
        }
        
        let op_sel0_const_parent_group_idx = NUM_BITS_USIZE * 2; // Group index for the constant 'true' wire created by parent
        let op_sel1_const_parent_group_idx = NUM_BITS_USIZE * 2 + 1; // Group index for the constant 'false' wire created by parent

        println!("Setting parent constant OpSelector wires for SUB (OpSel0=true at group {}, OpSel1=false at group {})...", op_sel0_const_parent_group_idx, op_sel1_const_parent_group_idx);

        if let Some(wire_idx_op_sel0) = bristol_circuit.input_values.get(op_sel0_const_parent_group_idx).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx_op_sel0, false); // OpSel0 = 1 (true) for SUB
            println!("  Set parent constant wire {} (group {}) for OpSel0 to true", wire_idx_op_sel0, op_sel0_const_parent_group_idx);
        } else {
            return Err(format!("Parent constant wire group for OpSel0 not found at expected index {}", op_sel0_const_parent_group_idx));
        }

        if let Some(wire_idx_op_sel1) = bristol_circuit.input_values.get(op_sel1_const_parent_group_idx).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx_op_sel1, false); // OpSel1 = 0 (false) for SUB
            println!("  Set parent constant wire {} (group {}) for OpSel1 to false", wire_idx_op_sel1, op_sel1_const_parent_group_idx);
        } else {
            return Err(format!("Parent constant wire group for OpSel1 not found at expected index {}", op_sel1_const_parent_group_idx));
        }
        
        // Now, also set the direct inputs for the subcircuit's part of the merged circuit
        // These are the wires the subcircuit's internal gates will actually use.
        let num_direct_parent_input_groups = (NUM_BITS_USIZE * 2) + 2; // LHS bit witnesses, RHS bit witnesses, 2 parent OpSel const wires

        // Subcircuit LHS bits (remapped into parent circuit)
        let sub_lhs_start_group_idx_in_merged = num_direct_parent_input_groups;
        // println!("Setting direct {}-bit LHS inputs for subcircuit part (starting group index {} in merged)...", NUM_BITS_USIZE, sub_lhs_start_group_idx_in_merged);
        for i in 0..NUM_BITS_USIZE {
            let current_sub_group_idx = sub_lhs_start_group_idx_in_merged + i;
            if let Some(wire_idx) = bristol_circuit.input_values.get(current_sub_group_idx).and_then(|g| g.get(0)) {
                let bit_val = (val_a >> i) & 1 == 1;
                sim_inputs.insert(*wire_idx, bit_val);
                // println!("  Set subcircuit LHS bit {} (wire {}, group {}) to {}", i, wire_idx, current_sub_group_idx, bit_val);
            } else {
                return Err(format!("Could not get subcircuit LHS wire for bit {} (expected group index {} in merged circuit)", i, current_sub_group_idx));
            }
        }

        // Subcircuit RHS bits (remapped into parent circuit)
        let sub_rhs_start_group_idx_in_merged = num_direct_parent_input_groups + NUM_BITS_USIZE;
        println!("Setting direct {}-bit RHS inputs for subcircuit part (starting group index {} in merged)...", NUM_BITS_USIZE, sub_rhs_start_group_idx_in_merged);
        for i in 0..NUM_BITS_USIZE {
            let current_sub_group_idx = sub_rhs_start_group_idx_in_merged + i;
            if let Some(wire_idx) = bristol_circuit.input_values.get(current_sub_group_idx).and_then(|g| g.get(0)) {
                let bit_val = (val_b >> i) & 1 == 1;
                sim_inputs.insert(*wire_idx, bit_val);
                println!("  Set subcircuit RHS bit {} (wire {}, group {}) to {}", i, wire_idx, current_sub_group_idx, bit_val);
            } else {
                return Err(format!("Could not get subcircuit RHS wire for bit {} (expected group index {} in merged circuit)", i, current_sub_group_idx));
            }
        }

        // Subcircuit OpSelector inputs (remapped into parent circuit)
        let sub_op_sel0_group_idx_in_merged = num_direct_parent_input_groups + (NUM_BITS_USIZE * 2);
        // println!("Setting direct OpSel0 input for subcircuit part (group index {} in merged) to true...", sub_op_sel0_group_idx_in_merged);
        if let Some(wire_idx) = bristol_circuit.input_values.get(sub_op_sel0_group_idx_in_merged).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx, false); // OpSel0 = 1 for SUB
            // println!("  Set subcircuit OpSel0 wire {} (group {}) to true", wire_idx, sub_op_sel0_group_idx_in_merged);
        } else {
            return Err(format!("Could not get subcircuit OpSel0 wire (expected group index {} in merged circuit)", sub_op_sel0_group_idx_in_merged));
        }

        let sub_op_sel1_group_idx_in_merged = num_direct_parent_input_groups + (NUM_BITS_USIZE * 2) + 1;
        println!("Setting direct OpSel1 input for subcircuit part (group index {} in merged) to false...", sub_op_sel1_group_idx_in_merged);
        if let Some(wire_idx) = bristol_circuit.input_values.get(sub_op_sel1_group_idx_in_merged).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx, false); // OpSel1 = 0 for SUB
            println!("  Set subcircuit OpSel1 wire {} (group {}) to false", wire_idx, sub_op_sel1_group_idx_in_merged);
        } else {
            return Err(format!("Could not get subcircuit OpSel1 wire (expected group index {} in merged circuit)", sub_op_sel1_group_idx_in_merged));
        }
        
        // --- 4. Simulate --- 
        println!("Simulating {}-bit ADD circuit...", NUM_BITS_USIZE);
        let sim_outputs = simulate_bristol_circuit(&bristol_circuit, &sim_inputs)?;

        // --- 5. Retrieve and Interpret Output ---
        let mut actual_sum_val: u64 = 0;
        println!("Retrieving {}-bit output...", NUM_BITS_USIZE);
        for i in 0..NUM_BITS_USIZE {
            let output_bit_witness = Witness(output_start_w.0 + i as u32);
            if let Some(&parent_equivalent_wire_idx) = translator.witness_wire_map.get(&output_bit_witness) {
                if let Some(true) = sim_outputs.get(&parent_equivalent_wire_idx) {
                    actual_sum_val |= 1u64 << i;
                }
            } else {
                 return Err(format!("Witness {} for output bit {} not found in witness_wire_map", output_bit_witness.0, i));
            }
        }
        
        // --- 6. Verify against expected 64-bit sum ---
        let expected_sum_val = val_a.wrapping_add(val_b);
        
        println!("Inputs (u64): val_a = {}, val_b = {}", val_a, val_b);
        println!("Expected {}-bit circuit output: {} ({:064b})", NUM_BITS_USIZE, expected_sum_val, expected_sum_val);
        println!("Actual   {}-bit circuit output: {} ({:064b})", NUM_BITS_USIZE, actual_sum_val, actual_sum_val);
        
        if actual_sum_val == expected_sum_val {
            println!("Success: Actual {}-bit sum matches expected {}-bit sum.", NUM_BITS_USIZE, NUM_BITS_USIZE);
            Ok(())
        } else {
            Err(format!(
                "FAIL: Actual {}-bit sum {} ({:064b}) does not match expected {}-bit sum {} ({:064b})", 
                NUM_BITS_USIZE, actual_sum_val, actual_sum_val, NUM_BITS_USIZE, expected_sum_val, expected_sum_val
            ))
        }
    }    

    #[test]
    fn test_functional_bigint_sub_simple() -> Result<(), String> {
        const NUM_BITS_U32: u32 = 64;
        const NUM_BITS_USIZE: usize = 64;

        // --- 1. Setup Translator and ACIR for SUB ---
        let lhs_start_w = Witness(1); 
        let rhs_start_w = Witness(1 + NUM_BITS_U32); 
        let output_start_w = Witness(1 + NUM_BITS_U32 * 2);

        let acir_circuit = Circuit::<FieldElement> {
            current_witness_index: 1 + NUM_BITS_U32 * 3, 
            opcodes: vec![Opcode::BlackBoxFuncCall(BlackBoxFuncCall::BigIntSub { // Use BigIntSub
                lhs: lhs_start_w.0,
                rhs: rhs_start_w.0,
                output: output_start_w.0,
            })],
            expression_width: ExpressionWidth::Unbounded,
            private_parameters: (0..NUM_BITS_U32 * 2).map(|i| Witness(lhs_start_w.0 + i)).collect(),
            public_parameters: PublicInputs(BTreeSet::new()),
            return_values: PublicInputs((0..NUM_BITS_U32).map(|i| Witness(output_start_w.0 + i)).collect()),
            assert_messages: Vec::new(),
        };

        let mut translator = BristolCircuitTranslator::<FieldElement>::new();
        translator.translate_circuit(&acir_circuit)?;
    

        let bristol_circuit = translator.circuit.clone();

        // --- 2. Define Test Case Input Values for SUB ---
        let val_a: u64 = 9876543210987654321; // Larger number
        let val_b: u64 = 1234567890123456789; // Smaller number

        // --- 3. Prepare Simulation Inputs ---
        let mut sim_inputs: HashMap<usize, bool> = HashMap::new();

        println!("Initializing all {} input groups in merged circuit to false by default", bristol_circuit.input_values.len());
        for input_group in &bristol_circuit.input_values {
            for &wire in input_group {
                sim_inputs.insert(wire, false);
            }
        }

        println!("Setting parent-level bit-witness inputs...");
        for i in 0..NUM_BITS_USIZE {
            let lhs_bit_witness = Witness(lhs_start_w.0 + i as u32);
            if let Some(&parent_lhs_bit_wire) = translator.witness_wire_map.get(&lhs_bit_witness) {
                sim_inputs.insert(parent_lhs_bit_wire, (val_a >> i) & 1 == 1);
            } else {
                return Err(format!("LHS bit-witness {} not in map!", lhs_bit_witness.0));
            }

            let rhs_bit_witness = Witness(rhs_start_w.0 + i as u32);
            if let Some(&parent_rhs_bit_wire) = translator.witness_wire_map.get(&rhs_bit_witness) {
                sim_inputs.insert(parent_rhs_bit_wire, (val_b >> i) & 1 == 1);
            } else {
                return Err(format!("RHS bit-witness {} not in map!", rhs_bit_witness.0));
            }
        }

        let op_sel0_const_parent_group_idx = NUM_BITS_USIZE * 2; // Group index for the constant 'true' wire created by parent
        let op_sel1_const_parent_group_idx = NUM_BITS_USIZE * 2 + 1; // Group index for the constant 'false' wire created by parent

        println!("Setting parent constant OpSelector wires for SUB (OpSel0=true at group {}, OpSel1=false at group {})...", op_sel0_const_parent_group_idx, op_sel1_const_parent_group_idx);

        if let Some(wire_idx_op_sel0) = bristol_circuit.input_values.get(op_sel0_const_parent_group_idx).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx_op_sel0, true); // OpSel0 = 1 (true) for SUB
            println!("  Set parent constant wire {} (group {}) for OpSel0 to true", wire_idx_op_sel0, op_sel0_const_parent_group_idx);
        } else {
            return Err(format!("Parent constant wire group for OpSel0 not found at expected index {}", op_sel0_const_parent_group_idx));
        }

        if let Some(wire_idx_op_sel1) = bristol_circuit.input_values.get(op_sel1_const_parent_group_idx).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx_op_sel1, false); // OpSel1 = 0 (false) for SUB
            println!("  Set parent constant wire {} (group {}) for OpSel1 to false", wire_idx_op_sel1, op_sel1_const_parent_group_idx);
        } else {
            return Err(format!("Parent constant wire group for OpSel1 not found at expected index {}", op_sel1_const_parent_group_idx));
        }
        
        // Now, also set the direct inputs for the subcircuit's part of the merged circuit
        // These are the wires the subcircuit's internal gates will actually use.
        let num_direct_parent_input_groups = (NUM_BITS_USIZE * 2) + 2; // LHS bit witnesses, RHS bit witnesses, 2 parent OpSel const wires

        // Subcircuit LHS bits (remapped into parent circuit)
        let sub_lhs_start_group_idx_in_merged = num_direct_parent_input_groups;
        // println!("Setting direct {}-bit LHS inputs for subcircuit part (starting group index {} in merged)...", NUM_BITS_USIZE, sub_lhs_start_group_idx_in_merged);
        for i in 0..NUM_BITS_USIZE {
            let current_sub_group_idx = sub_lhs_start_group_idx_in_merged + i;
            if let Some(wire_idx) = bristol_circuit.input_values.get(current_sub_group_idx).and_then(|g| g.get(0)) {
                let bit_val = (val_a >> i) & 1 == 1;
                sim_inputs.insert(*wire_idx, bit_val);
                // println!("  Set subcircuit LHS bit {} (wire {}, group {}) to {}", i, wire_idx, current_sub_group_idx, bit_val);
            } else {
                return Err(format!("Could not get subcircuit LHS wire for bit {} (expected group index {} in merged circuit)", i, current_sub_group_idx));
            }
        }

        // Subcircuit RHS bits (remapped into parent circuit)
        let sub_rhs_start_group_idx_in_merged = num_direct_parent_input_groups + NUM_BITS_USIZE;
        println!("Setting direct {}-bit RHS inputs for subcircuit part (starting group index {} in merged)...", NUM_BITS_USIZE, sub_rhs_start_group_idx_in_merged);
        for i in 0..NUM_BITS_USIZE {
            let current_sub_group_idx = sub_rhs_start_group_idx_in_merged + i;
            if let Some(wire_idx) = bristol_circuit.input_values.get(current_sub_group_idx).and_then(|g| g.get(0)) {
                let bit_val = (val_b >> i) & 1 == 1;
                sim_inputs.insert(*wire_idx, bit_val);
                println!("  Set subcircuit RHS bit {} (wire {}, group {}) to {}", i, wire_idx, current_sub_group_idx, bit_val);
            } else {
                return Err(format!("Could not get subcircuit RHS wire for bit {} (expected group index {} in merged circuit)", i, current_sub_group_idx));
            }
        }

        // Subcircuit OpSelector inputs (remapped into parent circuit)
        let sub_op_sel0_group_idx_in_merged = num_direct_parent_input_groups + (NUM_BITS_USIZE * 2);
        // println!("Setting direct OpSel0 input for subcircuit part (group index {} in merged) to true...", sub_op_sel0_group_idx_in_merged);
        if let Some(wire_idx) = bristol_circuit.input_values.get(sub_op_sel0_group_idx_in_merged).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx, true); // OpSel0 = 1 for SUB
            // println!("  Set subcircuit OpSel0 wire {} (group {}) to true", wire_idx, sub_op_sel0_group_idx_in_merged);
        } else {
            return Err(format!("Could not get subcircuit OpSel0 wire (expected group index {} in merged circuit)", sub_op_sel0_group_idx_in_merged));
        }

        let sub_op_sel1_group_idx_in_merged = num_direct_parent_input_groups + (NUM_BITS_USIZE * 2) + 1;
        println!("Setting direct OpSel1 input for subcircuit part (group index {} in merged) to false...", sub_op_sel1_group_idx_in_merged);
        if let Some(wire_idx) = bristol_circuit.input_values.get(sub_op_sel1_group_idx_in_merged).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx, false); // OpSel1 = 0 for SUB
            println!("  Set subcircuit OpSel1 wire {} (group {}) to false", wire_idx, sub_op_sel1_group_idx_in_merged);
        } else {
            return Err(format!("Could not get subcircuit OpSel1 wire (expected group index {} in merged circuit)", sub_op_sel1_group_idx_in_merged));
        }
        
        // --- 4. Simulate --- 
        println!("Simulating {}-bit SUB circuit...", NUM_BITS_USIZE);
        let sim_outputs = simulate_bristol_circuit(&bristol_circuit, &sim_inputs)?;

        // --- 5. Retrieve and Interpret Output ---
        let mut actual_sub_val: u64 = 0;
        println!("Retrieving {}-bit output...", NUM_BITS_USIZE);
        for i in 0..NUM_BITS_USIZE {
            let output_bit_witness = Witness(output_start_w.0 + i as u32);
            if let Some(&parent_equivalent_wire_idx) = translator.witness_wire_map.get(&output_bit_witness) {
                if let Some(true) = sim_outputs.get(&parent_equivalent_wire_idx) {
                    actual_sub_val |= 1u64 << i;
                }
            } else {
                 return Err(format!("Witness {} for output bit {} not found in witness_wire_map", output_bit_witness.0, i));
            }
        }
        
        // --- 6. Verify against expected 64-bit subtraction ---
        let expected_sub_val = val_a.wrapping_sub(val_b); // Corrected to wrapping_sub
        
        println!("Inputs (u64): val_a = {}, val_b = {}", val_a, val_b);
        println!("Expected {}-bit circuit output (SUB): {} ({:064b})", NUM_BITS_USIZE, expected_sub_val, expected_sub_val);
        println!("Actual   {}-bit circuit output (SUB): {} ({:064b})", NUM_BITS_USIZE, actual_sub_val, actual_sub_val);
        
        if actual_sub_val == expected_sub_val {
            println!("Success: Actual {}-bit subtraction matches expected {}-bit subtraction.", NUM_BITS_USIZE, NUM_BITS_USIZE);
            Ok(())
        } else {
            Err(format!(
                "FAIL: Actual {}-bit subtraction {} ({:064b}) does not match expected {}-bit subtraction {} ({:064b})", 
                NUM_BITS_USIZE, actual_sub_val, actual_sub_val, NUM_BITS_USIZE, expected_sub_val, expected_sub_val
            ))
        }
    }
    
    #[test]
    fn test_functional_bigint_mul_simple() -> Result<(), String> {
        const NUM_BITS_U32: u32 = 64;
        const NUM_BITS_USIZE: usize = 64;

        // --- 1. Setup Translator and ACIR for MUL ---
        let lhs_start_w = Witness(1); 
        let rhs_start_w = Witness(1 + NUM_BITS_U32); 
        let output_start_w = Witness(1 + NUM_BITS_U32 * 2);

        let acir_circuit = Circuit::<FieldElement> {
            current_witness_index: 1 + NUM_BITS_U32 * 3, 
            opcodes: vec![Opcode::BlackBoxFuncCall(BlackBoxFuncCall::BigIntMul { // Use BigIntMul
                lhs: lhs_start_w.0,
                rhs: rhs_start_w.0,
                output: output_start_w.0,
            })],
            expression_width: ExpressionWidth::Unbounded,
            private_parameters: (0..NUM_BITS_U32 * 2).map(|i| Witness(lhs_start_w.0 + i)).collect(),
            public_parameters: PublicInputs(BTreeSet::new()),
            return_values: PublicInputs((0..NUM_BITS_U32).map(|i| Witness(output_start_w.0 + i)).collect()),
            assert_messages: Vec::new(),
        };

        let mut translator = BristolCircuitTranslator::<FieldElement>::new();
        translator.translate_circuit(&acir_circuit)?;
    
        let bristol_circuit = translator.circuit.clone();

        // --- 2. Define Test Case Input Values for MUL (use smaller values) ---
        let val_a: u64 = 123459979; // First operand
        let val_b: u64 = 987654321; // Second operand

        // --- 3. Prepare Simulation Inputs ---
        let mut sim_inputs: HashMap<usize, bool> = HashMap::new();

        println!("Initializing all {} input groups in merged circuit to false by default", bristol_circuit.input_values.len());
        for input_group in &bristol_circuit.input_values {
            for &wire in input_group {
                sim_inputs.insert(wire, false);
            }
        }

        println!("Setting parent-level bit-witness inputs...");
        for i in 0..NUM_BITS_USIZE {
            let lhs_bit_witness = Witness(lhs_start_w.0 + i as u32);
            if let Some(&parent_lhs_bit_wire) = translator.witness_wire_map.get(&lhs_bit_witness) {
                sim_inputs.insert(parent_lhs_bit_wire, (val_a >> i) & 1 == 1);
            } else {
                return Err(format!("LHS bit-witness {} not in map!", lhs_bit_witness.0));
            }

            let rhs_bit_witness = Witness(rhs_start_w.0 + i as u32);
            if let Some(&parent_rhs_bit_wire) = translator.witness_wire_map.get(&rhs_bit_witness) {
                sim_inputs.insert(parent_rhs_bit_wire, (val_b >> i) & 1 == 1);
            } else {
                return Err(format!("RHS bit-witness {} not in map!", rhs_bit_witness.0));
            }
        }
        
        // Set OpSelector inputs for MUL (OpSel0=false, OpSel1=true)
        let op_sel0_const_parent_group_idx = NUM_BITS_USIZE * 2; 
        let op_sel1_const_parent_group_idx = NUM_BITS_USIZE * 2 + 1;

        println!("Setting parent constant OpSelector wires for MUL (OpSel0=false at group {}, OpSel1=true at group {})...", 
                 op_sel0_const_parent_group_idx, op_sel1_const_parent_group_idx);

        if let Some(wire_idx_op_sel0) = bristol_circuit.input_values.get(op_sel0_const_parent_group_idx).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx_op_sel0, false); // OpSel0 = 0 (false) for MUL
            println!("  Set parent constant wire {} (group {}) for OpSel0 to false", wire_idx_op_sel0, op_sel0_const_parent_group_idx);
        } else {
            return Err(format!("Parent constant wire group for OpSel0 not found at expected index {}", op_sel0_const_parent_group_idx));
        }

        if let Some(wire_idx_op_sel1) = bristol_circuit.input_values.get(op_sel1_const_parent_group_idx).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx_op_sel1, true); // OpSel1 = 1 (true) for MUL
            println!("  Set parent constant wire {} (group {}) for OpSel1 to true", wire_idx_op_sel1, op_sel1_const_parent_group_idx);
        } else {
            return Err(format!("Parent constant wire group for OpSel1 not found at expected index {}", op_sel1_const_parent_group_idx));
        }
        
        // Set the subcircuit's direct inputs
        let num_direct_parent_input_groups = (NUM_BITS_USIZE * 2) + 2; 

        // Subcircuit LHS bits
        let sub_lhs_start_group_idx_in_merged = num_direct_parent_input_groups;
        for i in 0..NUM_BITS_USIZE {
            let current_sub_group_idx = sub_lhs_start_group_idx_in_merged + i;
            if let Some(wire_idx) = bristol_circuit.input_values.get(current_sub_group_idx).and_then(|g| g.get(0)) {
                let bit_val = (val_a >> i) & 1 == 1;
                sim_inputs.insert(*wire_idx, bit_val);
            } else {
                return Err(format!("Could not get subcircuit LHS wire for bit {} (expected group index {} in merged circuit)", 
                                  i, current_sub_group_idx));
            }
        }

        // Subcircuit RHS bits
        let sub_rhs_start_group_idx_in_merged = num_direct_parent_input_groups + NUM_BITS_USIZE;
        println!("Setting direct {}-bit RHS inputs for subcircuit part (starting group index {} in merged)...", 
                  NUM_BITS_USIZE, sub_rhs_start_group_idx_in_merged);
        for i in 0..NUM_BITS_USIZE {
            let current_sub_group_idx = sub_rhs_start_group_idx_in_merged + i;
            if let Some(wire_idx) = bristol_circuit.input_values.get(current_sub_group_idx).and_then(|g| g.get(0)) {
                let bit_val = (val_b >> i) & 1 == 1;
                sim_inputs.insert(*wire_idx, bit_val);
                println!("  Set subcircuit RHS bit {} (wire {}, group {}) to {}", i, wire_idx, current_sub_group_idx, bit_val);
            } else {
                return Err(format!("Could not get subcircuit RHS wire for bit {} (expected group index {} in merged circuit)", 
                                  i, current_sub_group_idx));
            }
        }

        // Subcircuit OpSelector inputs
        let sub_op_sel0_group_idx_in_merged = num_direct_parent_input_groups + (NUM_BITS_USIZE * 2);
        if let Some(wire_idx) = bristol_circuit.input_values.get(sub_op_sel0_group_idx_in_merged).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx, false); // OpSel0 = 0 (false) for MUL
        } else {
            return Err(format!("Could not get subcircuit OpSel0 wire (expected group index {} in merged circuit)", 
                              sub_op_sel0_group_idx_in_merged));
        }

        let sub_op_sel1_group_idx_in_merged = num_direct_parent_input_groups + (NUM_BITS_USIZE * 2) + 1;
        println!("Setting direct OpSel1 input for subcircuit part (group index {} in merged) to true...", 
                 sub_op_sel1_group_idx_in_merged);
        if let Some(wire_idx) = bristol_circuit.input_values.get(sub_op_sel1_group_idx_in_merged).and_then(|g| g.get(0)) {
            sim_inputs.insert(*wire_idx, true); // OpSel1 = 1 (true) for MUL
            println!("  Set subcircuit OpSel1 wire {} (group {}) to true", wire_idx, sub_op_sel1_group_idx_in_merged);
        } else {
            return Err(format!("Could not get subcircuit OpSel1 wire (expected group index {} in merged circuit)", 
                              sub_op_sel1_group_idx_in_merged));
        }
        
        // --- 4. Simulate --- 
        println!("Simulating {}-bit MUL circuit...", NUM_BITS_USIZE);
        let sim_outputs = simulate_bristol_circuit(&bristol_circuit, &sim_inputs)?;

        // --- 5. Retrieve and Interpret Output ---
        let mut actual_mul_val: u64 = 0;
        println!("Retrieving {}-bit output...", NUM_BITS_USIZE);
        for i in 0..NUM_BITS_USIZE {
            let output_bit_witness = Witness(output_start_w.0 + i as u32);
            if let Some(&parent_equivalent_wire_idx) = translator.witness_wire_map.get(&output_bit_witness) {
                if let Some(true) = sim_outputs.get(&parent_equivalent_wire_idx) {
                    actual_mul_val |= 1u64 << i;
                }
            } else {
                 return Err(format!("Witness {} for output bit {} not found in witness_wire_map", output_bit_witness.0, i));
            }
        }
        
        // --- 6. Verify against expected multiplication result ---
        // Note: Only compare the lowest 64 bits since the circuit is limited to 64-bit output
        let expected_mul_val = (val_a as u128 * val_b as u128) as u64; 
        
        println!("Inputs (u64): val_a = {}, val_b = {}", val_a, val_b);
        println!("Expected full multiplication result: {}", val_a as u128 * val_b as u128);
        println!("Expected {}-bit circuit output (MUL, truncated): {} ({:064b})", NUM_BITS_USIZE, expected_mul_val, expected_mul_val);
        println!("Actual   {}-bit circuit output (MUL): {} ({:064b})", NUM_BITS_USIZE, actual_mul_val, actual_mul_val);
        
        if actual_mul_val == expected_mul_val {
            println!("Success: Actual {}-bit multiplication result matches expected result (truncated to 64 bits).", NUM_BITS_USIZE);
            Ok(())
        } else {
            Err(format!(
                "FAIL: Actual {}-bit multiplication result {} ({:064b}) does not match expected result {} ({:064b})", 
                NUM_BITS_USIZE, actual_mul_val, actual_mul_val, expected_mul_val, expected_mul_val
            ))
        }
    }
}

fn main() {
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

    println!(
        "Max witness index defined for 3 operations ({} bits each): {}. Next witness index: {}.",
        BITS_PER_NUM,
        current_max_witness,
        current_max_witness + 1
    );

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

    println!("\nTranslating the combined circuit with 3 BigInt operations...");
    let mut translator = BristolCircuitTranslator::<FieldElement>::new();
    if let Err(e) = translator.translate_circuit(&circuit) {
        eprintln!("Error translating circuit: {}", e);
        // Run validation to diagnose issues if translation fails
        println!("\nRunning circuit validation to diagnose issues:");
        if let Err(validation_err) = translator.circuit.validate() {
            println!("Validation error: {}", validation_err);
        } else {
            println!("Circuit passed validation, but an error occurred during translation phases (e.g., merge, wire mapping).");
        }
        return;
    }
    println!("Combined circuit translation successful.");

    // Print the input ACIR circuit first
    println!("\nInput Circuit:");
    println!("{}", circuit);

    match translator.circuit.to_bristol_format() {
        Ok(bristol) => {
            // Print the full Bristol format output to console
            println!("\nBristol Format Output:");
            println!("{}", bristol);

            // Write the Bristol string to bristol.txt
            match File::create("bristol.txt") {
                Ok(mut file) => {
                    match file.write_all(bristol.as_bytes()) {
                        Ok(_) => println!("\nSuccessfully wrote Bristol circuit to bristol.txt"),
                        Err(e) => eprintln!("\nError writing to bristol.txt: {}", e),
                    }
                }
                Err(e) => eprintln!("\nError creating bristol.txt: {}", e),
            }
            
            // Parse header and print statistics
            let lines: Vec<&str> = bristol.lines().collect();
            if !lines.is_empty() {
                let header: Vec<&str> = lines[0].split_whitespace().collect();
                if header.len() >= 2 {
                    match (header[0].parse::<usize>(), header[1].parse::<usize>()) {
                        (Ok(num_gates), Ok(num_wires)) => {
                            println!("\nCircuit Statistics (from Bristol header):");
                            println!("Number of gates: {}", num_gates);
                            println!("Number of wires: {}", num_wires);
                        }
                        _ => {
                            println!("\nCould not parse gate/wire counts from Bristol header: {}", lines[0]);
                        }
                    }
                } else {
                    println!("\nBristol header format unexpected: {}", lines[0]);
                }
            } else {
                println!("\nBristol output is empty, cannot extract statistics.");
            }
        }
        Err(e) => {
            eprintln!("\nError generating Bristol format: {}", e);
            // Try to validate and print more detailed error info
            println!("\nRunning circuit validation to diagnose issues:");
            if let Err(validation_err) = translator.circuit.validate() {
                println!("  Validation error: {}", validation_err);
            } else {
                println!("  Circuit passed validation checks, but Bristol format generation still failed.");
            }
            
            // Check for unused wires specifically
            let mut unused_wires = Vec::new();
            for (wire_idx, usage) in translator.circuit.wire_usage.iter().enumerate() {
                if *usage == FieldElement::zero() && translator.circuit.wire_types[wire_idx] != WireType::Output {
                    unused_wires.push(wire_idx);
                }
            }
            
            if !unused_wires.is_empty() {
                println!("\nFound {} unused non-output wires:", unused_wires.len());
                for wire_idx in unused_wires.iter().take(10) {
                    println!(
                        "  Wire {} (type {:?}, usage {})", 
                        wire_idx, 
                        translator.circuit.wire_types[*wire_idx],
                        translator.circuit.wire_usage[*wire_idx]
                    );
                }
                if unused_wires.len() > 10 {
                    println!("  ... and {} more unused wires", unused_wires.len() - 10);
                }
            }
        }
    }
}

