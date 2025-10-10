#!/bin/bash
#SBATCH --job-name=analyze_extractors
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/analyze_results_%j.out
#SBATCH --error=slurm_logs/analyze_results_%j.err

# SLURM script for analyzing feature extractor results
# This runs after all training jobs complete

echo "Starting analysis job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "============================================"

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment
source .venv/bin/activate

# Set display variables for headless operation
export DISPLAY=""
export MPLBACKEND="Agg"
export SDL_VIDEODRIVER="dummy"

# Combine results from individual runs
echo "Combining individual training results..."
python -c "
import json
from pathlib import Path
import sys

# Collect results from individual extractor runs
combined_results = {
    'comparison_metadata': {
        'combined_from_individual_runs': True,
        'individual_run_dirs': []
    },
    'results': {}
}

results_dir = Path('results')
individual_dirs = list(results_dir.glob('single_extractor_*'))

print(f'Found {len(individual_dirs)} individual result directories')

for individual_dir in individual_dirs:
    extractor_name = individual_dir.name.replace('single_extractor_', '')
    results_file = individual_dir / 'complete_results.json'
    
    if results_file.exists():
        print(f'Loading results for {extractor_name}')
        with open(results_file) as f:
            data = json.load(f)
            
        # Add to combined results
        if 'results' in data and extractor_name in data['results']:
            combined_results['results'][extractor_name] = data['results'][extractor_name]
            combined_results['comparison_metadata']['individual_run_dirs'].append(str(individual_dir))
        else:
            print(f'Warning: No valid results found for {extractor_name}')
    else:
        print(f'Warning: Results file not found for {extractor_name}: {results_file}')

if combined_results['results']:
    # Save combined results
    combined_dir = Path('results/combined_analysis')
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    combined_file = combined_dir / 'complete_results.json'
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f'Combined results saved to: {combined_file}')
    print(f'Found results for: {list(combined_results[\"results\"].keys())}')
else:
    print('No results found to combine!')
    sys.exit(1)
"

# Check if combination was successful
COMBINED_RESULTS="results/combined_analysis/complete_results.json"
if [ -f "$COMBINED_RESULTS" ]; then
    echo "Running statistical analysis on combined results..."
    python scripts/analyze_feature_extractors.py "$COMBINED_RESULTS"
    
    if [ $? -eq 0 ]; then
        echo "Analysis completed successfully!"
        echo "Results available in: results/combined_analysis/analysis/"
    else
        echo "Analysis failed!"
    fi
else
    echo "Combined results file not created!"
    exit 1
fi

echo "============================================"
echo "Analysis job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"