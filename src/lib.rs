//! # burn-es — Evolution Strategies for Burn
//!
//! | arXiv | Function | What |
//! |-------|----------|------|
//! | [1703.03864](https://arxiv.org/abs/1703.03864) | `antithetic_pair` | Paired (+eps,-eps) — variance reduction |
//! | [2511.16652](https://arxiv.org/abs/2511.16652) | `eggroll_mutate` | Low-rank perturbation — 100x speedup |
//!
//! Weight mutation and fitness ranking for black-box optimization of
//! neural network parameters. Used for ternary weight hill-climbing,
//! GRPO policy search, and ES fine-tuning.
use burn::tensor::{backend::Backend, Distribution, Tensor};

/// Gaussian noise mutation.
pub fn gaussian_mutate<B: Backend>(
    w: Tensor<B, 2>,
    sigma: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let noise = Tensor::random(w.dims(), Distribution::Normal(0.0, sigma as f64), device);
    w + noise
}

/// Ternary weight mutation: perturb + re-ternarize.
///
/// Steps: sign(W-mean(W)) * mean(|W|).
pub fn ternary_mutate<B: Backend>(w: Tensor<B, 2>, sigma: f32, device: &B::Device) -> Tensor<B, 2> {
    let perturbed = gaussian_mutate(w, sigma, device);
    let scale = perturbed.clone().abs().mean().unsqueeze_dims(&[0, 0]);
    let mean = perturbed.clone().mean().unsqueeze_dims(&[0, 0]);
    perturbed.sub(mean).sign().mul(scale)
}

/// Low-rank perturbation via A·B^T (EGGROLL, 2511.16652).
///
/// ΔW = A · B^T where A ∈ R^{d_out × rank}, B ∈ R^{d_in × rank}.
/// rank ≪ min(d_out, d_in) — 100x fewer random numbers than full-rank.
pub fn eggroll_mutate<B: Backend>(
    w: Tensor<B, 2>,
    rank: usize,
    sigma: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let [d_out, d_in] = w.dims();
    let a = Tensor::random(
        [d_out, rank],
        Distribution::Normal(0.0, sigma as f64),
        device,
    );
    let b = Tensor::random(
        [d_in, rank],
        Distribution::Normal(0.0, sigma as f64),
        device,
    );
    w + a.matmul(b.transpose())
}

/// Antithetic sampling: paired +e/-e mutations (OpenAI ES, 1703.03864).
///
/// Returns `(w + noise, w - noise)`. Correlating errors reduces
/// gradient variance by ~2x without extra population members.
pub fn antithetic_pair<B: Backend>(
    w: Tensor<B, 2>,
    sigma: f32,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let noise = Tensor::random(w.dims(), Distribution::Normal(0.0, sigma as f64), device);
    (w.clone().add(noise.clone()), w.sub(noise))
}

/// Rank-based fitness normalization (OpenAI ES).
///
/// Replaces raw reward values with rank scores: `(rank_i - 1) / (N - 1)`.
/// Robust to reward scale outliers and non-stationarity.
///
/// Returns normalized values in `[0, 1]`. Higher reward → higher rank → higher score.
/// Ties get their average rank.
pub fn rank_fitness(rewards: &[f32]) -> Vec<f32> {
    let n = rewards.len();
    if n <= 1 {
        return rewards.to_vec();
    }
    let mut indexed: Vec<(usize, f32)> = rewards.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f32; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && (indexed[j + 1].1 - indexed[i].1).abs() < 1e-8 {
            j += 1;
        }
        let avg_rank = (i + j) as f32 / 2.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank / (n - 1).max(1) as f32;
        }
        i = j + 1;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};
    type B = NdArray;
    fn dev() -> NdArrayDevice {
        NdArrayDevice::default()
    }

    #[test]
    fn ternary_shape() {
        let w = Tensor::<B, 2>::random([64, 32], Distribution::Default, &dev());
        assert_eq!(ternary_mutate(w, 0.1, &dev()).dims(), [64, 32]);
    }
    #[test]
    fn eggroll_shape() {
        let w = Tensor::<B, 2>::random([32, 16], Distribution::Default, &dev());
        assert_eq!(eggroll_mutate(w, 4, 0.1, &dev()).dims(), [32, 16]);
    }
    #[test]
    fn antithetic_shape() {
        let w = Tensor::<B, 2>::ones([8, 4], &dev());
        let (p, n) = antithetic_pair(w, 0.1, &dev());
        assert_eq!(p.dims(), [8, 4]);
        assert_eq!(n.dims(), [8, 4]);
    }
    #[test]
    fn rank_fitness_range() {
        let r = rank_fitness(&[5.0, 1.0, 3.0, 10.0]);
        assert!(r.iter().all(|x| (0.0..=1.0).contains(x)));
        assert!((r[3] - 1.0).abs() < 0.01); // best = 1.0
        assert!((r[1] - 0.0).abs() < 0.01); // worst = 0.0
    }
}
