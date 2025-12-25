import sys
from pathlib import Path
from collections import Counter
import numpy as np

# Add project root to path
notebook_dir = Path.cwd()
if notebook_dir.name == 'notebooks':
    project_root = notebook_dir.parent
else:
    project_root = notebook_dir

sys.path.insert(0, str(project_root))

from datasets import load_from_disk

print("=== VALIDATION SPLIT FAIRNESS ANALYSIS ===\n")

# Load the dataset with both splits
print("Loading dataset...")
ds = load_from_disk("data/processed/hf")
print("Dataset loaded successfully!")

print("Dataset splits:")
print(f"  Train: {len(ds['train'])} samples")
print(f"  Validation: {len(ds['validation'])} samples")
print(f"  Total: {len(ds['train']) + len(ds['validation'])} samples")
print(f"  Train/Val ratio: {len(ds['train'])/len(ds['validation']):.2f}")

# Expected ratio (if using 0.1 validation ratio from process_data.py)
expected_val_ratio = 0.1
expected_train_samples = len(ds['train']) + len(ds['validation'])
expected_val_samples = int(expected_train_samples * expected_val_ratio)
print(f"  Expected validation samples (10%): {expected_val_samples}")
print(f"  Actual validation samples: {len(ds['validation'])}")
print()

def analyze_split_fairness(train_ds, val_ds, column_name, label):
    """Analyze if a column is fairly distributed between train and validation."""
    train_counts = Counter(train_ds[column_name])
    val_counts = Counter(val_ds[column_name])
    
    # Get all unique values
    all_values = set(train_counts.keys()) | set(val_counts.keys())
    
    print(f"\n{label.upper()} DISTRIBUTION:")
    print("-" * 50)
    print(f"{'Category':<20} {'Train':<8} {'Val':<8} {'Train%':<8} {'Val%':<8} {'Ratio':<8}")
    print("-" * 50)
    
    total_train = len(train_ds)
    total_val = len(val_ds)
    
    chi_square = 0
    for value in sorted(all_values):
        train_count = train_counts.get(value, 0)
        val_count = val_counts.get(value, 0)
        
        train_pct = train_count / total_train * 100 if total_train > 0 else 0
        val_pct = val_count / total_val * 100 if total_val > 0 else 0
        
        # Calculate expected vs actual for chi-square test
        expected_val = total_val * (train_count + val_count) / (total_train + total_val)
        if expected_val > 0:
            chi_square += (val_count - expected_val) ** 2 / expected_val
        
        ratio = val_pct / train_pct if train_pct > 0 else float('inf')
        
        print(f"{str(value)[:19]:<20} {train_count:<8} {val_count:<8} {train_pct:<8.1f} {val_pct:<8.1f} {ratio:<8.2f}")
    
    return chi_square

# Analyze category1 (gender) distribution
train_ds = ds['train']
val_ds = ds['validation']

chi_square_gender = analyze_split_fairness(train_ds, val_ds, 'category1', 'Gender (Category1)')

# Analyze category2 distribution (top 10 categories)
def get_top_categories(dataset, column, top_n=10):
    """Get top N categories from a dataset column."""
    counts = Counter(dataset[column])
    return [cat for cat, _ in counts.most_common(top_n)]

top_categories = get_top_categories(train_ds, 'category2', 10)
print(f"\nAnalyzing top {len(top_categories)} categories: {top_categories}")

# Filter datasets to only include top categories for detailed analysis
train_filtered = train_ds.filter(lambda x: x['category2'] in top_categories)
val_filtered = val_ds.filter(lambda x: x['category2'] in top_categories)

chi_square_categories = analyze_split_fairness(train_filtered, val_filtered, 'category2', 'Top Categories (Category2)')

# Analyze text lengths
def analyze_text_lengths(train_ds, val_ds):
    """Analyze text length distributions."""
    train_lengths = [len(str(x['text'])) for x in train_ds if x['text']]
    val_lengths = [len(str(x['text'])) for x in val_ds if x['text']]
    
    print("TEXT LENGTH ANALYSIS:")
    print("-" * 30)
    print(f"Train - Mean: {np.mean(train_lengths):.1f}, Std: {np.std(train_lengths):.1f}, Min: {min(train_lengths)}, Max: {max(train_lengths)}")
    print(f"Val   - Mean: {np.mean(val_lengths):.1f}, Std: {np.std(val_lengths):.1f}, Min: {min(val_lengths)}, Max: {max(val_lengths)}")
    
    # Kolmogorov-Smirnov test for distribution similarity
    from scipy import stats
    ks_stat, ks_p = stats.ks_2samp(train_lengths, val_lengths)
    print(f"KS test (distribution similarity): statistic={ks_stat:.3f}, p-value={ks_p:.3f}")
    print(f"Distributions are {'similar' if ks_p > 0.05 else 'different'} (p > 0.05)")

try:
    analyze_text_lengths(train_ds, val_ds)
except ImportError:
    print("\nText length analysis requires scipy. Install with: pip install scipy")

# Overall fairness assessment
print("=== FAIRNESS ASSESSMENT ===")
print(f"Gender distribution χ²: {chi_square_gender:.2f}")
print(f"Category distribution χ²: {chi_square_categories:.2f}")

# Interpret chi-square values (rough guideline)
def interpret_chi_square(chi_square, df):
    """Rough interpretation of chi-square values."""
    # For 2xN tables, df = N-1
    if chi_square < df:
        return "Excellent (very similar distributions)"
    elif chi_square < 2 * df:
        return "Good (similar distributions)"
    elif chi_square < 3 * df:
        return "Fair (some differences)"
    else:
        return "Poor (significant differences)"

gender_df = len(set(Counter(train_ds['category1']).keys()) | set(Counter(val_ds['category1']).keys())) - 1
categories_df = len(top_categories) - 1

print(f"Gender split quality: {interpret_chi_square(chi_square_gender, gender_df)}")
print(f"Category split quality: {interpret_chi_square(chi_square_categories, categories_df)}")

# Recommendations
print("=== RECOMMENDATIONS ===")
if chi_square_gender > 2 * gender_df or chi_square_categories > 2 * categories_df:
    print("⚠️  Consider re-splitting the dataset for better balance")
    print("   Use: ds.train_test_split(test_size=0.1, stratify_by_column='category1')")
else:
    print("✅ Validation split appears fair and balanced")