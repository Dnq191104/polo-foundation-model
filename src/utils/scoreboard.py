"""
Step 7 Scoreboard Utilities

Computes deltas between baseline and checkpoint metrics for Step 7 evaluation.
Focuses on weak classes, rare materials, and neckline performance.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class Step7Scoreboard:
    """
    Computes and formats Step 7 success metrics vs baseline.
    
    Success targets:
    - Neckline@10: +5 to +10 points
    - Category@10 for cardigans/shorts: +10 points
    - Material@10 for denim/leather: +10 points
    """
    
    # Success thresholds (percentage points)
    NECKLINE_MIN_GAIN = 5.0
    NECKLINE_TARGET_GAIN = 10.0
    WEAK_CLASS_MIN_GAIN = 10.0
    RARE_MAT_MIN_GAIN = 10.0
    
    # Weak classes to track
    WEAK_CLASSES = ['cardigans', 'shorts']
    
    # Rare materials to track
    RARE_MATERIALS = ['denim', 'leather']
    
    def __init__(self, baseline_metrics_path: str):
        """
        Initialize scoreboard with baseline metrics.
        
        Args:
            baseline_metrics_path: Path to baseline metrics JSON (from Step 6)
        """
        self.baseline_metrics_path = Path(baseline_metrics_path)
        self.baseline_metrics = self._load_metrics(baseline_metrics_path)
    
    def _load_metrics(self, metrics_path: str) -> Dict[str, Any]:
        """Load metrics JSON from eval_retrieval.py output."""
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return data.get('metrics', data)  # Handle both wrapped and direct format
    
    def compute_deltas(
        self,
        checkpoint_metrics_path: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Compute deltas between checkpoint and baseline.
        
        Args:
            checkpoint_metrics_path: Path to checkpoint metrics JSON
            top_k: Top-k value for metrics
            
        Returns:
            Dict with deltas and success indicators
        """
        checkpoint_metrics = self._load_metrics(checkpoint_metrics_path)
        
        baseline = self.baseline_metrics['overall']
        checkpoint = checkpoint_metrics['overall']
        
        # Overall metrics
        neckline_key = f'neckline_match@{top_k}'
        material_key = f'material_match@{top_k}_known_only'
        category_key = f'category_match@{top_k}'
        
        deltas = {
            'overall': {
                'neckline': self._compute_delta(
                    baseline.get(neckline_key, 0),
                    checkpoint.get(neckline_key, 0)
                ),
                'material_known': self._compute_delta(
                    baseline.get(material_key, 0),
                    checkpoint.get(material_key, 0)
                ),
                'category': self._compute_delta(
                    baseline.get(category_key, 0),
                    checkpoint.get(category_key, 0)
                ),
            },
            'weak_classes': {},
            'rare_materials': {},
            'success': {},
        }
        
        # Weak classes
        baseline_cat = self.baseline_metrics.get('by_category', {})
        checkpoint_cat = checkpoint_metrics.get('by_category', {})
        
        for cat in self.WEAK_CLASSES:
            if cat in baseline_cat and cat in checkpoint_cat:
                deltas['weak_classes'][cat] = self._compute_delta(
                    baseline_cat[cat].get(category_key, 0),
                    checkpoint_cat[cat].get(category_key, 0)
                )
        
        # Rare materials
        baseline_mat = self.baseline_metrics.get('by_material', {})
        checkpoint_mat = checkpoint_metrics.get('by_material', {})
        
        for mat in self.RARE_MATERIALS:
            if mat in baseline_mat and mat in checkpoint_mat:
                deltas['rare_materials'][mat] = self._compute_delta(
                    baseline_mat[mat].get(material_key, 0),
                    checkpoint_mat[mat].get(material_key, 0)
                )
        
        # Success criteria
        deltas['success'] = self._evaluate_success(deltas)
        
        return deltas
    
    def _compute_delta(self, baseline_val: float, checkpoint_val: float) -> Dict[str, float]:
        """Compute delta with absolute and percentage change."""
        baseline_pct = baseline_val * 100
        checkpoint_pct = checkpoint_val * 100
        delta_pct = checkpoint_pct - baseline_pct
        
        return {
            'baseline': baseline_pct,
            'checkpoint': checkpoint_pct,
            'delta': delta_pct,
        }
    
    def _evaluate_success(self, deltas: Dict[str, Any]) -> Dict[str, bool]:
        """Evaluate success criteria."""
        success = {}
        
        # Neckline
        neckline_delta = deltas['overall']['neckline']['delta']
        success['neckline_min'] = neckline_delta >= self.NECKLINE_MIN_GAIN
        success['neckline_target'] = neckline_delta >= self.NECKLINE_TARGET_GAIN
        
        # Weak classes
        weak_deltas = [
            deltas['weak_classes'][cat]['delta']
            for cat in self.WEAK_CLASSES
            if cat in deltas['weak_classes']
        ]
        success['weak_classes_all_improved'] = all(d > 0 for d in weak_deltas)
        success['weak_classes_min_gain'] = all(d >= self.WEAK_CLASS_MIN_GAIN for d in weak_deltas)
        
        # Rare materials
        rare_deltas = [
            deltas['rare_materials'][mat]['delta']
            for mat in self.RARE_MATERIALS
            if mat in deltas['rare_materials']
        ]
        success['rare_materials_all_improved'] = all(d > 0 for d in rare_deltas)
        success['rare_materials_min_gain'] = all(d >= self.RARE_MAT_MIN_GAIN for d in rare_deltas)
        
        # Overall success
        success['overall'] = (
            success.get('neckline_min', False) and
            success.get('weak_classes_all_improved', False) and
            success.get('rare_materials_all_improved', False)
        )
        
        return success
    
    def format_scoreboard(self, deltas: Dict[str, Any]) -> str:
        """Format scoreboard as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("STEP 7 SCOREBOARD")
        lines.append("=" * 60)
        lines.append("")
        
        # Overall metrics
        lines.append("OVERALL METRICS:")
        lines.append("-" * 40)
        
        for metric_name, metric_data in deltas['overall'].items():
            baseline = metric_data['baseline']
            checkpoint = metric_data['checkpoint']
            delta = metric_data['delta']
            
            symbol = "[OK]" if delta > 0 else "[FAIL]"
            lines.append(
                f"  {metric_name:20s} {baseline:6.1f}% -> {checkpoint:6.1f}% "
                f"({delta:+6.1f}) {symbol}"
            )
        
        lines.append("")
        
        # Weak classes
        if deltas['weak_classes']:
            lines.append("WEAK CLASSES (Category@10):")
            lines.append("-" * 40)
            
            for cat, metric_data in deltas['weak_classes'].items():
                baseline = metric_data['baseline']
                checkpoint = metric_data['checkpoint']
                delta = metric_data['delta']
                
                symbol = "[OK]" if delta >= self.WEAK_CLASS_MIN_GAIN else "[FAIL]"
                lines.append(
                    f"  {cat:20s} {baseline:6.1f}% -> {checkpoint:6.1f}% "
                    f"({delta:+6.1f}) {symbol}"
                )
            
            lines.append("")
        
        # Rare materials
        if deltas['rare_materials']:
            lines.append("RARE MATERIALS (Material@10):")
            lines.append("-" * 40)
            
            for mat, metric_data in deltas['rare_materials'].items():
                baseline = metric_data['baseline']
                checkpoint = metric_data['checkpoint']
                delta = metric_data['delta']
                
                symbol = "[OK]" if delta >= self.RARE_MAT_MIN_GAIN else "[FAIL]"
                lines.append(
                    f"  {mat:20s} {baseline:6.1f}% -> {checkpoint:6.1f}% "
                    f"({delta:+6.1f}) {symbol}"
                )
            
            lines.append("")
        
        # Success summary
        lines.append("SUCCESS CRITERIA:")
        lines.append("-" * 40)
        
        success = deltas['success']
        
        lines.append(f"  Neckline +5pt:       {self._format_bool(success.get('neckline_min', False))}")
        lines.append(f"  Neckline +10pt:      {self._format_bool(success.get('neckline_target', False))}")
        lines.append(f"  Weak classes +10pt:  {self._format_bool(success.get('weak_classes_min_gain', False))}")
        lines.append(f"  Rare materials +10pt: {self._format_bool(success.get('rare_materials_min_gain', False))}")
        lines.append("")
        lines.append(f"  OVERALL SUCCESS:     {self._format_bool(success.get('overall', False))}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _format_bool(self, value: bool) -> str:
        """Format boolean as colored checkmark/cross."""
        return "[PASS]" if value else "[FAIL]"
    
    def save_scoreboard(
        self,
        checkpoint_metrics_path: str,
        output_dir: str,
        top_k: int = 10
    ) -> str:
        """
        Compute and save scoreboard to output directory.
        
        Args:
            checkpoint_metrics_path: Path to checkpoint metrics
            output_dir: Output directory for scoreboard files
            top_k: Top-k value
            
        Returns:
            Path to saved scoreboard text file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute deltas
        deltas = self.compute_deltas(checkpoint_metrics_path, top_k)
        
        # Save JSON
        json_path = output_dir / "scoreboard_deltas.json"
        with open(json_path, 'w') as f:
            json.dump(deltas, f, indent=2)
        
        # Format and save text
        scoreboard_text = self.format_scoreboard(deltas)
        text_path = output_dir / "scoreboard.txt"
        with open(text_path, 'w') as f:
            f.write(scoreboard_text)
        
        return str(text_path)


def create_scoreboard(baseline_metrics_path: str) -> Step7Scoreboard:
    """
    Factory function to create a Step7Scoreboard.
    
    Args:
        baseline_metrics_path: Path to baseline metrics JSON
        
    Returns:
        Configured Step7Scoreboard instance
    """
    return Step7Scoreboard(baseline_metrics_path)

