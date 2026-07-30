# burn-es — Evolution Strategies for Burn

[![CI](https://github.com/sehaxe/burn-es/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-es/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-es)](https://crates.io/crates/burn-es)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.21-orange.svg)](https://burn.dev)

Weight mutation and fitness ranking for black-box optimization of neural network parameters.
Ternary hill-climbing, low-rank EGGROLL perturbations, antithetic sampling.

> Papers:
> [OpenAI ES](https://arxiv.org/abs/1703.03864) (Salimans et al., 2017),
> [EGGROLL](https://arxiv.org/abs/2511.16652) (Sarkar et al., 2025).

## Install

```bash
cargo add burn-es
```

## Quick start

```rust
use burn_es::{ternary_mutate, eggroll_mutate, antithetic_pair, rank_fitness};

// Ternary weight mutation
let w_mutated = ternary_mutate(weight, 0.1, &device);

// Low-rank EGGROLL perturbation (100x faster than full-rank)
let w_eggroll = eggroll_mutate(weight, 16, 0.1, &device);

// Antithetic pair: variance reduction via (+eps, -eps)
let (w_plus, w_minus) = antithetic_pair(weight, 0.1, &device);

// Rank-based fitness normalization
let norm_fitness = rank_fitness(&raw_rewards);
```

## API

| Export | What |
|--------|------|
| `gaussian_mutate(w, s)` | Gaussian noise perturbation |
| `ternary_mutate(w, s)` | Perturb + re-ternarize |
| `eggroll_mutate(w, r, s)` | Low-rank A * B^T perturbation |
| `antithetic_pair(w, s)` | Paired (+eps, -eps) mutations |
| `rank_fitness(scores)` | Rank-based normalization |

## License

AGPL-3.0. See [LICENSE](LICENSE).
