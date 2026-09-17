//! Struktura DFA anomaly detector for the ESA-ADB TimeEval benchmark.
//!
//! Detrended Fluctuation Analysis (Peng et al., Phys. Rev. E, 1994) on sliding
//! windows. For each window the scaling exponent alpha is compared to a baseline
//! established from the first `baseline_fraction` of the series. A deviation
//! produces an anomaly score in [0, 1].
//!
//! Multivariate: each channel is scored independently; the per-timestamp score
//! is the maximum across channels (any-channel-anomalous semantics).
//!
//! Interface: JSON config as argv[1] (TimeEval convention):
//!   dataInput, dataOutput, customParameters, executionType
//!
//! First Rust-based algorithm in ESA-ADB.

use std::env;
use std::fs;
use std::io::Write;

use serde::Deserialize;

// ── DFA core (self-contained, same algorithm as struktura::dfa) ─────────────

fn ln(x: f64) -> f64 { libm::log(x) }
fn sqrt(x: f64) -> f64 { libm::sqrt(x) }
fn powf(x: f64, y: f64) -> f64 { libm::pow(x, y) }
fn powi(base: f64, exp: i32) -> f64 { libm::pow(base, exp as f64) }

struct DfaResult {
    alpha: f64,
    r_squared: f64,
}

fn linreg(x: &[f64], y: &[f64]) -> DfaResult {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
    let syy: f64 = y.iter().map(|v| v * v).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-15 {
        return DfaResult { alpha: 0.5, r_squared: 0.0 };
    }
    let slope = (n * sxy - sx * sy) / denom;
    let ss_tot = syy - sy * sy / n;
    let ss_res = syy - slope * sxy - (sy - slope * sx) * sy / n;
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    DfaResult { alpha: slope, r_squared: r2.max(0.0) }
}

fn dfa(values: &[f64]) -> DfaResult {
    let n = values.len();
    if n < 64 {
        return DfaResult { alpha: 0.5, r_squared: 0.0 };
    }
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let mut profile = Vec::with_capacity(n);
    let mut cum = 0.0;
    for &v in values {
        cum += v - mean;
        profile.push(cum);
    }
    let s_min = 16usize.max(n / 50);
    let s_max = n / 4;
    if s_min >= s_max {
        return DfaResult { alpha: 0.5, r_squared: 0.0 };
    }
    let ratio = powf(s_max as f64 / s_min as f64, 1.0 / 11.0);
    let mut log_s = [0.0f64; 12];
    let mut log_f = [0.0f64; 12];
    let mut pts = 0usize;
    let mut prev_s = 0usize;
    for step in 0..12 {
        let s = (s_min as f64 * powi(ratio, step)) as usize;
        if s == prev_s || s > s_max {
            continue;
        }
        prev_s = s;
        let num_segs = n / s;
        if num_segs == 0 {
            continue;
        }
        let k = s as f64;
        let sx = k * (k - 1.0) / 2.0;
        let sx2 = k * (k - 1.0) * (2.0 * k - 1.0) / 6.0;
        let det = k * sx2 - sx * sx;
        if det.abs() < 1e-15 {
            continue;
        }
        let mut f2_sum = 0.0;
        for seg in 0..num_segs {
            let start = seg * s;
            let (mut sy, mut sxy, mut sy2) = (0.0, 0.0, 0.0);
            for i in 0..s {
                let yi = profile[start + i];
                sy += yi;
                sxy += i as f64 * yi;
                sy2 += yi * yi;
            }
            let a0 = (sx2 * sy - sx * sxy) / det;
            let a1 = (k * sxy - sx * sy) / det;
            let resid = (sy2 - a0 * sy - a1 * sxy).max(0.0);
            f2_sum += resid / k;
        }
        let f = sqrt(f2_sum / num_segs as f64);
        if f > 0.0 {
            log_s[pts] = ln(s as f64);
            log_f[pts] = ln(f);
            pts += 1;
        }
    }
    if pts < 3 {
        return DfaResult { alpha: 0.5, r_squared: 0.0 };
    }
    linreg(&log_s[..pts], &log_f[..pts])
}

// ── NaN handling ────────────────────────────────────────────────────────────

/// Forward-fill NaN/Inf, then backward-fill leading NaNs. Returns false if
/// the entire series is non-finite.
fn sanitize(values: &mut [f64]) -> bool {
    let mut last = f64::NAN;
    for v in values.iter_mut() {
        if v.is_finite() {
            last = *v;
        } else if last.is_finite() {
            *v = last;
        }
    }
    let mut first = f64::NAN;
    for v in values.iter() {
        if v.is_finite() {
            first = *v;
            break;
        }
    }
    if !first.is_finite() {
        return false;
    }
    for v in values.iter_mut() {
        if !v.is_finite() {
            *v = first;
        } else {
            break;
        }
    }
    true
}

// ── Robust baseline (median + MAD) ──────────────────────────────────────────

fn median_of(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    } else {
        xs[n / 2]
    }
}

/// Returns (center, scale) using median and MAD * 1.4826 (consistency factor
/// for normal data). More resistant to early anomalies than mean + std.
fn robust_baseline(xs: &[f64]) -> (f64, f64) {
    if xs.len() < 2 {
        return (xs.first().copied().unwrap_or(0.5), 0.1);
    }
    let mut sorted = xs.to_vec();
    let center = median_of(&mut sorted);
    let mut deviations: Vec<f64> = xs.iter().map(|x| (x - center).abs()).collect();
    let scale = median_of(&mut deviations) * 1.4826;
    (center, scale.max(1e-10))
}

// ── Score one channel ───────────────────────────────────────────────────────

fn score_channel(
    values: &mut [f64],
    window: usize,
    stride: usize,
    baseline_frac: f64,
) -> Vec<f64> {
    let n = values.len();
    let mut scores = vec![0.0f64; n];

    if !sanitize(values) || n < window {
        return scores;
    }

    // Compute alpha on every sliding window.
    let mut alphas: Vec<(usize, f64)> = Vec::new();
    let mut pos = 0;
    while pos + window <= n {
        let r = dfa(&values[pos..pos + window]);
        if r.r_squared > 0.3 {
            alphas.push((pos + window / 2, r.alpha));
        }
        pos += stride;
    }

    if alphas.is_empty() {
        return scores;
    }

    // Robust baseline from the first baseline_fraction windows.
    let bl_count = ((alphas.len() as f64 * baseline_frac) as usize)
        .max(2)
        .min(alphas.len());
    let bl_alphas: Vec<f64> = alphas[..bl_count].iter().map(|(_, a)| *a).collect();
    let (bl_center, bl_scale) = robust_baseline(&bl_alphas);

    // Score: |alpha - center| / scale, sigmoid to [0, 1].
    for &(center, alpha) in &alphas {
        let z = ((alpha - bl_center).abs() / bl_scale).min(20.0);
        let score = 2.0 / (1.0 + libm::exp(-z)) - 1.0;
        let start = center.saturating_sub(window / 2);
        let end = (start + window).min(n);
        for i in start..end {
            if score > scores[i] {
                scores[i] = score;
            }
        }
    }

    scores
}

// ── TimeEval interface ──────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Config {
    data_input: String,
    data_output: String,
    #[serde(default)]
    custom_parameters: Params,
    #[serde(default)]
    execution_type: String,
}

#[derive(Deserialize)]
struct Params {
    #[serde(default = "default_window")]
    window_size: usize,
    #[serde(default = "default_stride")]
    stride: usize,
    #[serde(default = "default_baseline")]
    baseline_fraction: f64,
    #[serde(default)]
    random_state: u64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            window_size: default_window(),
            stride: default_stride(),
            baseline_fraction: default_baseline(),
            random_state: 42,
        }
    }
}

fn default_window() -> usize { 1024 }
fn default_stride() -> usize { 256 }
fn default_baseline() -> f64 { 0.2 }

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: struktura-timeeval '<json config>'");
        std::process::exit(2);
    }
    let config: Config = serde_json::from_str(&args[1]).unwrap_or_else(|e| {
        eprintln!("config parse error: {e}");
        std::process::exit(2);
    });

    if config.execution_type == "train" {
        eprintln!("struktura_dfa: unsupervised, no training phase");
        return;
    }

    // ── Parse CSV ───────────────────────────────────────────────────────
    let csv_text = fs::read_to_string(&config.data_input).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {e}", config.data_input);
        std::process::exit(1);
    });

    let mut lines_iter = csv_text.lines();
    let header = lines_iter.next().unwrap_or("");
    let columns: Vec<&str> = header.split(',').map(|s| s.trim()).collect();

    // Data columns: everything except "timestamp" and "is_anomaly*".
    let data_col_indices: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != "timestamp" && !c.starts_with("is_anomaly"))
        .map(|(i, _)| i)
        .collect();

    let n_channels = data_col_indices.len();
    if n_channels == 0 {
        eprintln!("struktura_dfa: no data columns found");
        std::process::exit(1);
    }

    let rows: Vec<Vec<&str>> = lines_iter.map(|l| l.split(',').collect()).collect();
    let n = rows.len();

    let mut channels: Vec<Vec<f64>> = (0..n_channels)
        .map(|ci| {
            let col = data_col_indices[ci];
            rows.iter()
                .map(|fields| {
                    fields
                        .get(col)
                        .and_then(|s| s.trim().parse::<f64>().ok())
                        .unwrap_or(f64::NAN)
                })
                .collect()
        })
        .collect();

    let window = config.custom_parameters.window_size.max(64);
    let stride = config.custom_parameters.stride.max(1);
    let baseline_frac = config.custom_parameters.baseline_fraction.clamp(0.05, 0.95);

    eprintln!(
        "struktura_dfa: {} samples x {} channel(s), window={}, stride={}, baseline={:.0}%",
        n, n_channels, window, stride, baseline_frac * 100.0
    );

    // ── Score each channel, take per-timestamp max ──────────────────────
    let mut final_scores = vec![0.0f64; n];

    for (ci, channel) in channels.iter_mut().enumerate() {
        let ch_scores = score_channel(channel, window, stride, baseline_frac);
        let ch_name = columns.get(data_col_indices[ci]).unwrap_or(&"?");
        let flagged = ch_scores.iter().filter(|&&s| s > 0.5).count();
        eprintln!("  channel {:>20}: {} flagged timestamps", ch_name, flagged);
        for (i, &s) in ch_scores.iter().enumerate() {
            if s > final_scores[i] {
                final_scores[i] = s;
            }
        }
    }

    // ── Write output ────────────────────────────────────────────────────
    let mut out = fs::File::create(&config.data_output).unwrap_or_else(|e| {
        eprintln!("cannot write {}: {e}", config.data_output);
        std::process::exit(1);
    });
    writeln!(out, "anomaly_score").unwrap();
    for s in &final_scores {
        writeln!(out, "{:.6}",  s).unwrap();
    }

    eprintln!("struktura_dfa: wrote {} scores to {}", n, config.data_output);
}
