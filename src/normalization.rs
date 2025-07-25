// normalization.rs
// Core normalization library for financial data
//
// This module provides robust normalization techniques specifically designed for financial data
// which often has non-Gaussian distributions (heavy tails, skewness, outliers).
//
// Key principles:
// - Prices: Usually log-normal, use z-score normalization after log transform if needed
// - Quantities: Heavy-tailed, power-law distributed, use quantile normalization
// - Never use arbitrary clipping - use mathematically sound transformations

use std::collections::HashMap;

/// Normalization strategy for different types of financial data
#[derive(Debug, Clone)]
pub enum NormalizationStrategy {
    /// Standard z-score normalization (mean=0, std=1) - for normally distributed data
    ZScore { mean: f64, std: f64 },
    /// Quantile normalization - maps empirical quantiles to standard normal quantiles
    /// Best for heavy-tailed, skewed distributions like orderbook quantities
    QuantileNormal { quantiles: Vec<f64> },
    /// Log transformation followed by z-score - for log-normal data like prices
    LogNormal { log_mean: f64, log_std: f64 },
    /// Robust normalization using median and IQR - for data with outliers
    Robust { median: f64, iqr: f64 },
}

/// Statistics for a data column
#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub q25: f64,
    pub q75: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub zero_count: usize,
    pub total_count: usize,
}

/// Normalization parameters for a complete dataset
///
/// RECOMMENDATION: Use quantile normalization for all financial data.
/// It's distribution-agnostic and always produces standard normal output.
#[derive(Debug, Clone)]
pub struct NormalizationParams {
    /// Default strategy for all columns
    pub default_strategy: NormalizationStrategy,
    /// Column-specific overrides (rarely needed)
    pub column_strategies: HashMap<String, NormalizationStrategy>,
}

impl NormalizationParams {
    /// Create normalization parameters using quantile normalization for everything
    /// This is the recommended approach for financial data.
    pub fn quantile_for_all() -> Self {
        Self {
            default_strategy: NormalizationStrategy::QuantileNormal { quantiles: Vec::new() },
            column_strategies: HashMap::new(),
        }
    }

    /// Create normalization parameters from data analysis (legacy method)
    /// DEPRECATED: Use quantile_for_all() instead for better consistency
    pub fn from_data_analysis(
        price_columns: &[String],
        quantity_columns: &[String],
        column_stats: &HashMap<String, ColumnStats>,
    ) -> Self {
        // Just use quantile normalization for everything - it's better
        Self::quantile_for_all()
    }

    /// Get the appropriate strategy for a column
    pub fn get_strategy(&self, column_name: &str) -> &NormalizationStrategy {
        self.column_strategies.get(column_name)
            .unwrap_or(&self.default_strategy)
    }
}

/// Compute detailed statistics for a column of data
pub fn compute_column_stats(values: &[f64]) -> ColumnStats {
    if values.is_empty() {
        return ColumnStats {
            mean: 0.0, std: 0.0, median: 0.0, min: 0.0, max: 0.0,
            q25: 0.0, q75: 0.0, skewness: 0.0, kurtosis: 0.0,
            zero_count: 0, total_count: 0,
        };
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    let median = sorted_values[n / 2];
    let q25 = sorted_values[n / 4];
    let q75 = sorted_values[3 * n / 4];
    let min = sorted_values[0];
    let max = sorted_values[n - 1];

    let skewness = if std > 1e-10 {
        values.iter().map(|&x| ((x - mean) / std).powi(3)).sum::<f64>() / n as f64
    } else {
        0.0
    };

    let kurtosis = if std > 1e-10 {
        values.iter().map(|&x| ((x - mean) / std).powi(4)).sum::<f64>() / n as f64
    } else {
        3.0
    };

    let zero_count = values.iter().filter(|&&x| x.abs() < 1e-10).count();

    ColumnStats {
        mean, std, median, min, max, q25, q75, skewness, kurtosis,
        zero_count, total_count: n,
    }
}

/// Create quantile normalization strategy from data
pub fn create_quantile_strategy(values: &[f64]) -> NormalizationStrategy {
    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Store quantiles for later lookup
    NormalizationStrategy::QuantileNormal {
        quantiles: sorted_values
    }
}

/// Create a complete normalization setup for financial data
/// This is the recommended way to set up normalization for any financial dataset
pub fn create_financial_normalization(
    price_data: &[f64],
    quantity_data: &[f64],
) -> (NormalizationStrategy, NormalizationStrategy) {
    let price_strategy = create_quantile_strategy(price_data);
    let quantity_strategy = create_quantile_strategy(quantity_data);
    (price_strategy, quantity_strategy)
}

/// Apply normalization to a single value
pub fn normalize_value(value: f64, strategy: &NormalizationStrategy) -> f64 {
    match strategy {
        NormalizationStrategy::ZScore { mean, std } => {
            if *std > 1e-10 {
                (value - mean) / std
            } else {
                0.0
            }
        }
        NormalizationStrategy::QuantileNormal { quantiles } => {
            if quantiles.is_empty() {
                return 0.0;
            }
            
            // Find quantile of this value
            let quantile = find_quantile(quantiles, value);
            // Map to standard normal
            inverse_normal_cdf(quantile as f32) as f64
        }
        NormalizationStrategy::LogNormal { log_mean, log_std } => {
            if value > 0.0 && *log_std > 1e-10 {
                (value.ln() - log_mean) / log_std
            } else {
                0.0
            }
        }
        NormalizationStrategy::Robust { median, iqr } => {
            if *iqr > 1e-10 {
                (value - median) / iqr
            } else {
                0.0
            }
        }
    }
}

/// Find the quantile (0-1) of a value in a sorted array
fn find_quantile(sorted_values: &[f64], target: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.5;
    }
    
    // Binary search for the position
    let pos = sorted_values.binary_search_by(|&x| x.partial_cmp(&target).unwrap())
        .unwrap_or_else(|x| x);
    
    (pos as f64 + 0.5) / sorted_values.len() as f64
}

/// Approximate inverse normal CDF (Beasley-Springer-Moro algorithm)
pub fn inverse_normal_cdf(p: f32) -> f32 {
    if p <= 0.0 { return -6.0; }
    if p >= 1.0 { return 6.0; }
    
    // Beasley-Springer-Moro approximation
    let a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00
    ];
    
    let b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01
    ];
    
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00
    ];
    
    let d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00
    ];
    
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    
    if p < p_low {
        // Rational approximation for lower region
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if p <= p_high {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        // Rational approximation for upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}
