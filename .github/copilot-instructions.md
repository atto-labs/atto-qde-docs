# Copilot Instructions for Atto

This project implements a **Quantum Decision Engine (QDE)** using a Python package named `atto`.

## Purpose

The system models how beliefs evolve into decisions under uncertainty using a quantum-inspired approach.

## Core Mental Model

```
context → belief state → evolving reasoning → decision
```

## Key Concepts

- **Superposition**: multiple strategies coexist
- **Interference**: signals influence each other
- **Non-commutativity**: order of signals matters
- **Measurement**: state collapses into a decision

## Architecture Rules

### Core vs Scenario Separation

- `atto/core/` contains domain-agnostic mathematical logic
- `atto/scenarios/` contains domain-specific mappings (e.g. energy)

### Naming Conventions

- Use "state", "operator", "measurement", "dynamics"
- Avoid generic ML terms like "model weights" unless necessary
- Prefer "belief state" over "feature vector"

### Design Principles

- Keep the core engine reusable across domains
- Scenario logic must not modify core math
- All decisions emerge from state evolution, not direct prediction

### API Philosophy

The system is a "decision engine", not a classifier. Use language like:

- "evolve state"
- "collapse to decision"
- "apply operator"

### Implementation Constraints

- Use numpy for core math (no heavy frameworks initially)
- Keep functions small and composable
- Prefer clarity over optimisation

### Documentation Style

- Explain code in terms of: "how beliefs evolve into decisions"
- Avoid purely mathematical explanations without intuition

## Goal

Ensure all generated code aligns with the QDE paradigm and remains extensible across domains.
