//! Struktura DFA anomaly detector for the ESA-ADB TimeEval benchmark.
//!
//! Detrended Fluctuation Analysis (Peng et al., Phys. Rev. E, 1994) on sliding
//! windows. For each window the scaling exponent alpha is compared to a baseline
//! established from the first `baseline_fraction` of the series. A window whose
//! robust z-score (|alpha - median| / (1.4826 * MAD)) exceeds `threshold` marks
//! its samples as anomalous.
//!
//! Output follows the ESA-ADB convention (as in the std algorithm): no header,
//! one 0/1 column per target channel, channels sorted by name. The evaluation
//! min-max scales each column and casts it to uint8, so only binary output
//! survives it.
//!
//! Interface: JSON config as argv[1] (TimeEval convention):
//!   dataInput, dataOutput, customParameters, executionType
//!
//! First Rust-based algorithm in ESA-ADB.

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Write};

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
    threshold: f64,
    causal: bool,
) -> Vec<u8> {
    let n = values.len();
    let mut flags = vec![0u8; n];
    let scores = &mut flags;

    if !sanitize(values) || n < window {
        return flags;
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
        return flags;
    }

    // Robust baseline from the first baseline_fraction windows.
    let bl_count = ((alphas.len() as f64 * baseline_frac) as usize)
        .max(2)
        .min(alphas.len());
    let bl_alphas: Vec<f64> = alphas[..bl_count].iter().map(|(_, a)| *a).collect();
    let (bl_center, bl_scale) = robust_baseline(&bl_alphas);

    // Flag the samples of a window whose robust z-score exceeds the threshold: the whole
    // window (offline), or, causally, the `stride` samples that follow the window: a streaming
    // detector decides when the window's last value arrives and marks only what comes after.
    for &(center, alpha) in &alphas {
        let z = (alpha - bl_center).abs() / bl_scale;
        if z > threshold {
            let w_end = (center.saturating_sub(window / 2) + window).min(n);
            let (start, end) = if causal { (w_end, (w_end + stride).min(n)) } else { (w_end.saturating_sub(window), w_end) };
            for flag in &mut scores[start..end] {
                *flag = 1;
            }
        }
    }

    flags
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
    #[serde(default = "default_threshold")]
    threshold: f64,
    #[serde(default)]
    target_channels: Option<Vec<String>>,
    #[serde(default)]
    causal: bool,
    #[serde(default)]
    random_state: u64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            window_size: default_window(),
            stride: default_stride(),
            baseline_fraction: default_baseline(),
            threshold: default_threshold(),
            target_channels: None,
            causal: false,
            random_state: 42,
        }
    }
}

fn default_window() -> usize { 1024 }
fn default_stride() -> usize { 256 }
fn default_baseline() -> f64 { 0.2 }
fn default_threshold() -> f64 { 3.5 }

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

    // ── Parse CSV (streamed: ESA-ADB test files have millions of rows) ─────
    let file = fs::File::open(&config.data_input).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {e}", config.data_input);
        std::process::exit(1);
    });
    let mut lines_iter = BufReader::new(file).lines().map_while(Result::ok);
    let header = lines_iter.next().unwrap_or_default();
    let columns: Vec<&str> = header.split(',').map(|s| s.trim()).collect();

    // Data columns: everything except "timestamp" and "is_anomaly*".
    let data_cols: Vec<&str> = columns
        .iter()
        .copied()
        .filter(|c| *c != "timestamp" && !c.starts_with("is_anomaly"))
        .collect();
    // Target channels as in the std algorithm: the requested ones that exist
    // (deduplicated), else all data columns; then sorted by name.
    let mut targets: Vec<&str> = Vec::new();
    for ch in config.custom_parameters.target_channels.iter().flatten() {
        if data_cols.contains(&ch.as_str()) && !targets.contains(&ch.as_str()) {
            targets.push(ch.as_str());
        }
    }
    if targets.is_empty() {
        targets = data_cols.clone();
    }
    targets.sort_unstable();
    let data_col_indices: Vec<usize> = targets
        .iter()
        .map(|t| columns.iter().position(|c| c == t).expect("target is a column"))
        .collect();

    let n_channels = data_col_indices.len();
    if n_channels == 0 {
        eprintln!("struktura_dfa: no data columns found");
        std::process::exit(1);
    }

    let mut channels: Vec<Vec<f64>> = vec![Vec::new(); n_channels];
    for line in lines_iter {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        for (ch, &col) in channels.iter_mut().zip(&data_col_indices) {
            ch.push(fields.get(col).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN));
        }
    }
    let n = channels[0].len();

    let window = config.custom_parameters.window_size.max(64);
    let stride = config.custom_parameters.stride.max(1);
    let baseline_frac = config.custom_parameters.baseline_fraction.clamp(0.05, 0.95);
    let threshold = config.custom_parameters.threshold;

    eprintln!(
        "struktura_dfa: {} samples x {} channel(s), window={}, stride={}, baseline={:.0}%, threshold={}",
        n, n_channels, window, stride, baseline_frac * 100.0, threshold
    );

    // ── Flag each target channel independently ──────────────────────────
    let mut flags: Vec<Vec<u8>> = Vec::with_capacity(n_channels);
    for (ci, channel) in channels.iter_mut().enumerate() {
        let ch = score_channel(channel, window, stride, baseline_frac, threshold, config.custom_parameters.causal);
        let flagged = ch.iter().filter(|&&f| f == 1).count();
        eprintln!("  channel {:>20}: {} flagged timestamps", targets[ci], flagged);
        flags.push(ch);
    }

    // ── Write output ────────────────────────────────────────────────────
    let out = fs::File::create(&config.data_output).unwrap_or_else(|e| {
        eprintln!("cannot write {}: {e}", config.data_output);
        std::process::exit(1);
    });
    let mut out = std::io::BufWriter::new(out);
    let mut line = String::new();
    for i in 0..n {
        line.clear();
        for (ci, ch) in flags.iter().enumerate() {
            if ci > 0 {
                line.push(',');
            }
            line.push(if ch[i] == 1 { '1' } else { '0' });
        }
        writeln!(out, "{line}").unwrap();
    }

    eprintln!("struktura_dfa: wrote {} rows x {} channel(s) to {}", n, n_channels, config.data_output);
}
