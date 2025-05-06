# Project ACIR2BRISTOL

## Setup 

### 1. Clone the repository

First, clone the repository `Noir`:

```bash
git clone git@github.com:noir-lang/noir.git
cd A
```

### 2. 
```bash
cp ./bristol_circuit ./noir/acvm-repo/acir/tests
```

Update ./noir/acvm-repo/acir/Cargo.toml 
```
...
[[bin]]
name = "bristol_circuit"
path = "tests/bristol_circuit.rs"
```

### 3. Build and Test

```bash
cargo build
cargo test
```
