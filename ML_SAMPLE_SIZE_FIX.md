# ML Sample Size Issue - Fixed ✅

## Problem
When creating ML features, only very few samples were being generated because:
- The feature engineering was **always** computing moving averages for ALL periods `[5, 10, 20, 50, 100, 200]`
- A 200-day moving average creates 200 NaN values at the start
- All NaN rows were dropped, leaving very few valid samples
- Example: 500 days of data → only ~280 valid samples after dropping NaN

## Solution

### 1. Respect User-Selected Lookback Periods
**File**: `quant_framework/ml/features.py`

**Before**:
```python
# Always used all periods
for period in [5, 10, 20, 50, 100, 200]:
    df[f'sma_{period}'] = TechnicalIndicators.sma(close, period)
```

**After**:
```python
# Use only user-selected periods + basic [5, 10, 20]
ma_periods = sorted(set(self.lookback_periods + [5, 10, 20]))
for period in ma_periods:
    if period <= len(df):  # Check data availability
        df[f'sma_{period}'] = TechnicalIndicators.sma(close, period)
```

### 2. Better Defaults
**Changed default lookback periods from** `[5, 10, 20, 50]` **to** `[5, 10, 20]`

This preserves more samples by default while still providing meaningful features.

### 3. Enhanced User Feedback
Added real-time feedback showing:
- Initial data rows loaded
- Valid samples after feature engineering
- Number and percentage of dropped rows
- Warnings when >50% of samples are lost
- Recommendations to use smaller lookback periods

### 4. Visual Metrics
Shows 4-column metric display:
- 📊 **Features**: Number of features created
- 📈 **Valid Samples**: Usable training samples
- 🗑️ **Dropped Rows**: How many rows lost (with %)
- 🎯 **Target Distribution**: Class balance or mean

## How to Use Effectively

### For Maximum Samples (Recommended for <500 days of data)
```
Lookback Periods: [5, 10, 20]
Result: ~20-30 rows dropped, 95%+ sample retention
```

### For Balanced Approach (500-1000 days)
```
Lookback Periods: [5, 10, 20, 50]
Result: ~50-60 rows dropped, 90%+ retention
```

### For Long-term Features (1000+ days)
```
Lookback Periods: [5, 10, 20, 50, 100]
Result: ~100-110 rows dropped, 85%+ retention
```

### Advanced Users with Multi-year Data (2000+ days)
```
Lookback Periods: [5, 10, 20, 50, 100, 200]
Result: ~200-210 rows dropped, still 90%+ retention
```

## Examples

### Example 1: SPY (2020-2024, ~1000 days)
**Old behavior**: 100, 200 always used → ~250 samples lost
**New behavior with [5, 10, 20]**: ~20 samples lost ✅

### Example 2: AAPL (2023-2024, ~250 days)
**Old behavior**: 200-day MA impossible, many NaN → ~50 samples
**New behavior with [5, 10, 20]**: ~230 samples ✅

## UI Improvements

### Info Box at Top
```
💡 Tip: Larger lookback periods (100, 200) create more features 
but lose more samples. For datasets <500 days, stick to [5, 10, 20] 
for best results.
```

### Sample Loss Warnings
- **>50% loss**: ⚠️ Warning with recommendation
- **30-50% loss**: ℹ️ Info that it's normal
- **<30% loss**: ✓ Success message

### Enhanced Help Text
The multiselect now shows:
```
"Periods for rolling calculations. Larger periods = fewer valid 
samples (e.g., 200-day MA loses first 200 rows)"
```

## Technical Details

### Why Samples Are Lost
1. **Rolling calculations** (moving averages, volatility) need historical data
2. **First N rows** become NaN where N = lookback period
3. **All features** must be valid (no NaN) for training
4. **Largest lookback** determines minimum valid row index

### Formula
```
Valid Samples ≈ Total Rows - max(lookback_periods) - forward_horizon
```

### What Changed
- ✅ Only compute MAs for user-selected periods
- ✅ Skip MA crossovers if periods not available
- ✅ Check data length before computing features
- ✅ Always include basic [5, 10, 20] for essential indicators
- ✅ Show transparent feedback on sample loss

## Best Practices

### 1. Start Small
Begin with `[5, 10, 20]` and add longer periods if needed

### 2. Match to Data Size
- 200-500 days → use [5, 10, 20]
- 500-1000 days → use [5, 10, 20, 50]
- 1000+ days → use [5, 10, 20, 50, 100]
- 2000+ days → use any periods

### 3. Check Sample Loss
Look at the feedback after creating features:
- If loss >50%, reduce lookback periods
- If loss <20%, you can safely add more periods

### 4. Balance Features vs Samples
More features ≠ better model. Quality > quantity.
- 30-50 features with 500 samples > 150 features with 150 samples

## Command Line Alternative

For programmatic use:
```python
from quant_framework.ml.features import FeatureEngineering

# Specify only what you need
fe = FeatureEngineering(
    lookback_periods=[5, 10, 20],  # Conservative
    include_technical=True,
    include_statistical=True,
    include_time=True,
    include_lagged=True
)

features = fe.create_features(data)
X, y = fe.create_training_data(features)
print(f"Created {len(X)} samples with {len(X.columns)} features")
```

## Result

✅ **No more "very few samples" issue!**
- Users have full control over feature periods
- Clear feedback on sample retention
- Smart defaults that work for most datasets
- Warnings when too many samples are lost
- Recommendations for optimization

**The ML Models tab now creates meaningful sample sizes for training!** 🎉

