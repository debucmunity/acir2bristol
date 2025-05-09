# Noir-to-Bristol for MPC

This project bridges [Noir](https://github.com/noir-lang/noir) with an existing Multi-Party Computation (MPC) framework by compiling Noir circuits into the **Bristol Fashion** format, enabling them to be executed in an MPC setting.

## What's this about?

MPC (Multi-Party Computation) is a cryptographic technique that allows multiple parties to jointly compute a function **without revealing their private inputs**. Unlike regular computation where everything happens transparently on a single machine, MPC keeps inputs and even intermediate steps private between participants.

To run an MPC protocol, we typically need a circuit — a description of the function we want to compute — and this circuit must be in a format that MPC tools can understand, like **Bristol Fashion**.

That’s where this project comes in.

## What we built

Our goal is to use **Noir** as a frontend language for writing MPC circuits. Noir is expressive, readable, and familiar for developers coming from a ZK background. However, most MPC frameworks don’t understand Noir directly.

So here’s what we did:

* Take Noir circuits.
* Compile them to ACIR (the intermediate format used by Noir).
* From ACIR, **translate into Bristol Fashion circuit format**.
* Finally, feed that Bristol circuit into [`mpc-framework`](https://github.com/debucmunity/mpc-framework) (forked from PSE), a lightweight MPC framework in TypeScript.

This lets us go from high-level Noir code ➝ MPC-ready circuit ➝ actual MPC run!

## Setup

Clone the repo and its submodules:

```bash
git clone --recurse-submodules https://github.com/debucmunity/acir2bristol
cd acir2bristol
```

Build and run:

```bash
cargo build
```

## NoirHack contribution

This project was built during **NoirHack**. While the idea of MPC and Bristol circuits isn’t new, our main contribution is:

* **Connecting Noir with MPC workflows** through ACIR.
* Building a working compiler/converter from **ACIR to Bristol format**.
* Integrating with [`mpc-framework`](https://github.com/debucmunity/mpc-framework) and successfully running example circuits.

It’s a step toward making Noir a more general-purpose frontend for secure computation, not just ZK!

## Future work

* Support more complex gates and optimizations in the converter.
* Add benchmarking + test suite for accuracy.
* Build a web frontend to write Noir and run MPC demos live.
