"""
Statistical analysis script for feature extractor comparison results.

This script analyzes the results from multi-extractor training and provides
statistical comparisons and visualizations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings('ignore')


class FeatureExtractorAnalyzer:
    """
    Analyzer for feature extractor comparison results.
    """
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer with results file.
        
        Args:
            results_file: Path to the complete_results.json file
        """
        self.results_file = Path(results_file)
        self.results_dir = self.results_file.parent
        
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.results = self.data['results']
        self.metadata = self.data['comparison_metadata']
        
        # Create analysis output directory
        self.analysis_dir = self.results_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with summary statistics for each extractor."""
        summary_data = []
        
        for name, result in self.results.items():
            if not result.get('completed', False):
                continue
                
            summary_data.append({
                'extractor_name': name,
                'extractor_type': result.get('extractor_type', 'unknown'),
                'best_reward': result.get('best_reward'),
                'final_reward': result.get('final_reward'),
                'total_parameters': result.get('total_parameters', 0),
                'trainable_parameters': result.get('trainable_parameters', 0),
                'training_time': result.get('training_time', 0),
                'timesteps_per_second': (
                    result.get('total_timesteps', 0) / result.get('training_time', 1)
                    if result.get('training_time', 0) > 0 else 0
                )
            })
        
        return pd.DataFrame(summary_data)
    
    def load_tensorboard_data(self, extractor_name: str) -> Optional[pd.DataFrame]:
        """
        Load tensorboard data for an extractor (if available).
        
        Args:
            extractor_name: Name of the extractor
            
        Returns:
            DataFrame with tensorboard metrics or None if not available
        """
        # This is a placeholder - in practice you'd use tensorboard's event file reader
        # For now, return None as tensorboard parsing requires additional dependencies
        return None
    
    def analyze_performance(self) -> Dict:
        """Analyze performance across different metrics."""
        df = self.create_summary_dataframe()
        
        if df.empty:
            return {"error": "No completed training runs found"}
        
        analysis = {
            "summary_statistics": {},
            "rankings": {},
            "statistical_tests": {}
        }
        
        # Summary statistics
        metrics = ['best_reward', 'final_reward', 'training_time', 'total_parameters']
        for metric in metrics:
            if metric in df.columns and df[metric].notna().any():
                analysis["summary_statistics"][metric] = {
                    "mean": float(df[metric].mean()),
                    "std": float(df[metric].std()),
                    "min": float(df[metric].min()),
                    "max": float(df[metric].max()),
                    "median": float(df[metric].median())
                }
        
        # Rankings
        if 'best_reward' in df.columns and df['best_reward'].notna().any():
            reward_ranking = df.nlargest(len(df), 'best_reward')[['extractor_name', 'best_reward']]
            analysis["rankings"]["by_reward"] = reward_ranking.to_dict('records')
        
        if 'total_parameters' in df.columns and df['total_parameters'].notna().any():
            param_ranking = df.nsmallest(len(df), 'total_parameters')[['extractor_name', 'total_parameters']]
            analysis["rankings"]["by_parameters"] = param_ranking.to_dict('records')
        
        if 'timesteps_per_second' in df.columns and df['timesteps_per_second'].notna().any():
            speed_ranking = df.nlargest(len(df), 'timesteps_per_second')[['extractor_name', 'timesteps_per_second']]
            analysis["rankings"]["by_speed"] = speed_ranking.to_dict('records')
        
        # Statistical tests (if we have enough data)
        if len(df) >= 2 and 'best_reward' in df.columns:
            analysis["statistical_tests"] = self._perform_statistical_tests(df)
        
        return analysis
    
    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict:
        """Perform statistical significance tests."""
        tests = {}
        
        # Since we typically have one run per extractor, we can't do traditional statistical tests
        # Instead, we'll provide descriptive statistics and effect sizes
        
        if len(df) >= 2:
            rewards = df['best_reward'].dropna()
            if len(rewards) >= 2:
                # Basic descriptive statistics
                tests["reward_analysis"] = {
                    "range": float(rewards.max() - rewards.min()),
                    "coefficient_of_variation": float(rewards.std() / rewards.mean()) if rewards.mean() != 0 else 0,
                    "best_vs_worst_ratio": float(rewards.max() / rewards.min()) if rewards.min() > 0 else float('inf')
                }
        
        return tests
    
    def create_visualizations(self) -> List[str]:
        """Create visualizations for the analysis."""
        df = self.create_summary_dataframe()
        
        if df.empty:
            print("No data available for visualization")
            return []
        
        created_plots = []
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Performance comparison bar chart
        if 'best_reward' in df.columns and df['best_reward'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(df['extractor_name'], df['best_reward'], 
                         color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink', 'lightgray'][:len(df)])
            ax.set_title('Best Reward by Feature Extractor', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Extractor', fontsize=12)
            ax.set_ylabel('Best Reward', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = self.analysis_dir / "reward_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(plot_path))
        
        # 2. Parameter efficiency scatter plot
        if all(col in df.columns for col in ['total_parameters', 'best_reward']):
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df['total_parameters'], df['best_reward'], 
                               s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
            
            for i, name in enumerate(df['extractor_name']):
                ax.annotate(name, (df.iloc[i]['total_parameters'], df.iloc[i]['best_reward']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax.set_title('Parameter Efficiency: Reward vs Model Size', fontsize=14, fontweight='bold')
            ax.set_xlabel('Total Parameters', fontsize=12)
            ax.set_ylabel('Best Reward', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.analysis_dir / "parameter_efficiency.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(plot_path))
        
        # 3. Training time comparison
        if 'training_time' in df.columns and df['training_time'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(df['extractor_name'], df['training_time'] / 60,  # Convert to minutes
                         color=['lightblue', 'lightgreen', 'lightsalmon', 'gold', 'plum', 'lightgray'][:len(df)])
            ax.set_title('Training Time by Feature Extractor', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Extractor', fontsize=12)
            ax.set_ylabel('Training Time (minutes)', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = self.analysis_dir / "training_time_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_plots.append(str(plot_path))
        
        # 4. Multi-metric comparison radar chart
        if len(df) > 1:
            try:
                metrics = ['best_reward', 'timesteps_per_second']
                available_metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
                
                if len(available_metrics) >= 2:
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                    
                    # Normalize metrics (higher is better)
                    df_norm = df[available_metrics].copy()
                    for col in available_metrics:
                        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
                    
                    # Parameters (lower is better, so invert)
                    if 'total_parameters' in df.columns and df['total_parameters'].notna().any():
                        df_norm['parameter_efficiency'] = 1 - ((df['total_parameters'] - df['total_parameters'].min()) / 
                                                             (df['total_parameters'].max() - df['total_parameters'].min()))
                        available_metrics.append('parameter_efficiency')
                    
                    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False)
                    angles = np.concatenate((angles, [angles[0]]))
                    
                    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
                    
                    for i, (_, row) in enumerate(df.iterrows()):
                        values = [df_norm.iloc[i][metric] for metric in available_metrics]
                        values = np.concatenate((values, [values[0]]))
                        
                        ax.plot(angles, values, 'o-', linewidth=2, label=row['extractor_name'], 
                               color=colors[i % len(colors)])
                        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
                    
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
                    ax.set_ylim(0, 1)
                    ax.set_title('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold', pad=20)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    
                    plt.tight_layout()
                    plot_path = self.analysis_dir / "multi_metric_comparison.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    created_plots.append(str(plot_path))
            except Exception as e:
                print(f"Could not create radar chart: {e}")
        
        return created_plots
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        analysis = self.analyze_performance()
        df = self.create_summary_dataframe()
        
        report_lines = [
            "# Feature Extractor Comparison Analysis Report",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Training Configuration",
            f"- Total timesteps per extractor: {self.metadata.get('total_timesteps_per_extractor', 'N/A'):,}",
            f"- Number of parallel environments: {self.metadata.get('n_envs', 'N/A')}",
            f"- Evaluation episodes: {self.metadata.get('n_eval_episodes', 'N/A')}",
            f"- Environment difficulty: {self.metadata.get('difficulty', 'N/A')}",
            "",
            "## Results Summary",
            ""
        ]
        
        if not df.empty:
            report_lines.extend([
                "### Performance Rankings",
                ""
            ])
            
            # Best reward ranking
            if 'by_reward' in analysis.get('rankings', {}):
                report_lines.append("**By Best Reward:**")
                for i, entry in enumerate(analysis['rankings']['by_reward'][:5], 1):
                    reward = entry.get('best_reward')
                    if reward is not None:
                        report_lines.append(f"{i}. {entry['extractor_name']}: {reward:.4f}")
                report_lines.append("")
            
            # Parameter efficiency
            if 'by_parameters' in analysis.get('rankings', {}):
                report_lines.append("**By Parameter Count (Lower is Better):**")
                for i, entry in enumerate(analysis['rankings']['by_parameters'][:5], 1):
                    params = entry.get('total_parameters')
                    if params is not None:
                        report_lines.append(f"{i}. {entry['extractor_name']}: {params:,} parameters")
                report_lines.append("")
            
            # Training speed
            if 'by_speed' in analysis.get('rankings', {}):
                report_lines.append("**By Training Speed (timesteps/second):**")
                for i, entry in enumerate(analysis['rankings']['by_speed'][:5], 1):
                    speed = entry.get('timesteps_per_second')
                    if speed is not None:
                        report_lines.append(f"{i}. {entry['extractor_name']}: {speed:.1f} timesteps/s")
                report_lines.append("")
            
            # Summary statistics table
            report_lines.extend([
                "### Detailed Results",
                "",
                "| Extractor | Type | Best Reward | Parameters | Training Time (min) |",
                "|-----------|------|-------------|------------|---------------------|"
            ])
            
            for _, row in df.iterrows():
                name = row['extractor_name']
                ext_type = row['extractor_type']
                reward = f"{row['best_reward']:.4f}" if pd.notna(row['best_reward']) else "N/A"
                params = f"{int(row['total_parameters']):,}" if pd.notna(row['total_parameters']) else "N/A"
                time_min = f"{row['training_time']/60:.1f}" if pd.notna(row['training_time']) else "N/A"
                
                report_lines.append(f"| {name} | {ext_type} | {reward} | {params} | {time_min} |")
        else:
            report_lines.append("No completed training runs found in the results.")
        
        report_lines.extend([
            "",
            "## Key Insights",
            ""
        ])
        
        # Generate insights based on analysis
        if not df.empty and 'best_reward' in df.columns:
            best_performer = df.loc[df['best_reward'].idxmax()]
            report_lines.append(f"- **Best performing extractor**: {best_performer['extractor_name']} with reward {best_performer['best_reward']:.4f}")
            
            if 'total_parameters' in df.columns:
                most_efficient = df.loc[df['total_parameters'].idxmin()]
                report_lines.append(f"- **Most parameter-efficient**: {most_efficient['extractor_name']} with {int(most_efficient['total_parameters']):,} parameters")
            
            if 'timesteps_per_second' in df.columns and df['timesteps_per_second'].notna().any():
                fastest = df.loc[df['timesteps_per_second'].idxmax()]
                report_lines.append(f"- **Fastest training**: {fastest['extractor_name']} at {fastest['timesteps_per_second']:.1f} timesteps/s")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = self.analysis_dir / "analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"Analysis report saved to: {report_path}")
        return report_text
    
    def run_complete_analysis(self) -> Dict:
        """Run complete analysis and generate all outputs."""
        print("Running feature extractor analysis...")
        
        # Perform analysis
        analysis = self.analyze_performance()
        
        # Create visualizations
        print("Creating visualizations...")
        plots = self.create_visualizations()
        
        # Generate report
        print("Generating report...")
        report = self.generate_report()
        
        # Save analysis results
        analysis_results = {
            "analysis": analysis,
            "plots_created": plots,
            "report_path": str(self.analysis_dir / "analysis_report.md"),
            "analysis_dir": str(self.analysis_dir)
        }
        
        analysis_path = self.analysis_dir / "analysis_results.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"Complete analysis saved to: {self.analysis_dir}")
        print(f"Created {len(plots)} visualization(s)")
        
        return analysis_results


def main():
    """Main function for running analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze feature extractor comparison results")
    parser.add_argument("results_file", help="Path to complete_results.json file")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Results file not found: {args.results_file}")
        return
    
    analyzer = FeatureExtractorAnalyzer(args.results_file)
    results = analyzer.run_complete_analysis()
    
    print("\nAnalysis completed!")
    print(f"Results directory: {results['analysis_dir']}")


if __name__ == "__main__":
    main()