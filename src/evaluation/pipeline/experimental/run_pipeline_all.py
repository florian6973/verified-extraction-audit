#!/usr/bin/env python3
from src._repo import REPO_ROOT
"""
Run the complete pipeline for all generation files in experimental-recall-all directory.
For each file, extracts parameters (model, dataset_size, n_epochs, pii_rate) and creates
a config file, then runs the full pipeline.
"""

import os
import re
import yaml
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional


def parse_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse generation filename to extract parameters.
    Pattern: generation_False_all_{dataset_size}_{model}_{pii_rate}_{n_epochs}_{budget}.parquet
    
    Returns:
        Dictionary with parsed parameters or None if pattern doesn't match
    """
    pattern = r'generation_False_all_(\d+)_(\d+B)_([\d.]+)_(\d+)_(\d+)\.csv'
    # pattern = r'generation_False_all_(\d+)_(\d+B)_([\d.]+)_(\d+)_(\d+)\.parquet'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    return {
        'dataset_size': int(match.group(1)),
        'model': match.group(2),
        'pii_rate': float(match.group(3)),
        'n_epochs': int(match.group(4)),
        'budget': int(match.group(5))
    }


def load_template_config(template_path: str) -> Dict:
    """Load the template config file."""
    with open(template_path, 'r') as f:
        return yaml.safe_load(f)


def create_config_from_template(template_config: Dict, params: Dict, output_config_path: str) -> str:
    """
    Create a new config file from template with updated parameters.
    
    Args:
        template_config: Template config dictionary
        params: Parameters to update (model, dataset_size, n_epochs, pii_rate, budget)
        output_config_path: Path to save the new config file
    
    Returns:
        Path to the created config file
    """
    # Create a copy of the template
    new_config = yaml.safe_load(yaml.dump(template_config))
    
    # Update filter parameters
    new_config['filters']['model'] = params['model']
    new_config['filters']['dataset_size'] = params['dataset_size']
    new_config['filters']['n_epochs'] = params['n_epochs']
    new_config['filters']['pii_rate'] = params['pii_rate']
    
    # Update budget in inputs
    new_config['inputs']['budget'] = params['budget']
    
    # Save the new config
    with open(output_config_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
    
    return output_config_path


def run_command(cmd: List[str], description: str, script_dir: str) -> bool:
    """
    Run a command and handle errors.
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            check=True,
            capture_output=False
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        return False


def run_pipeline_for_config(config_path: str, script_dir: str, plotting_only: bool = False) -> bool:
    """
    Run the complete pipeline for a given config file.
    
    Args:
        config_path: Path to config file
        script_dir: Directory containing the pipeline scripts
        plotting_only: If True, only run steps 9-13 (plotting steps)
    
    Returns:
        True if all steps succeeded, False otherwise
    """
    # pipeline-attack-5.py is one level up from experimental directory
    pipeline_dir = os.path.dirname(script_dir)
    pipeline_attack_script = os.path.join(pipeline_dir, 'pipeline-attack-5.py')
    
    commands = [
        # Step 1: Check names
        (['python', os.path.join(script_dir, 'check_names-improved.py'), '--config', config_path],
         'Step 1: Running check_names.py'),
        
        # Step 2: Extract names and compute ll for remaining values
        (['python', os.path.join(script_dir, 'ner_ll_remaining.py'), '--config', config_path],
         'Step 2: Running ner_ll_remaining.py'),
        
        # Step 3: Merge all names
        (['python', os.path.join(script_dir, 'merge_all_names.py'), '--config', config_path],
         'Step 3: Running merge_all_names.py'),
        
        # Step 4: Compute ll for all names
        (['python', os.path.join(script_dir, 'compute_ll_names.py'), '--config', config_path],
         'Step 4: Running compute_ll_names.py'),
        
        # Step 5: Prepare data for MIA
        (['python', os.path.join(script_dir, 'mia', 'prep_data.py'), '--config', config_path],
         'Step 5: Running mia/prep_data.py'),
        
        # Step 6: Train MIA verifier
        (['python', os.path.join(script_dir, 'mia', 'train_mia_verifier_cv.py'), 'train', '--config', config_path],
         'Step 6: Running mia/train_mia_verifier_cv.py'),
        
        # Step 7: Compute scores
        (['python', os.path.join(script_dir, 'mia', 'compute_scores.py'), '--config', config_path],
         'Step 7: Running mia/compute_scores.py'),
        
        # Step 8: Convert LL to probability
        (['python', os.path.join(script_dir, 'mia', 'll_to_prob.py'), '--config', config_path],
         'Step 8: Running mia/ll_to_prob.py'),
        
        # Step 9: Experimental evaluation
        (['python', os.path.join(script_dir, 'mia', 'evaluate_scores.py'), '--config', config_path],
         'Step 9: Running mia/evaluate_scores.py (Experimental Evaluation)'),
        
        # Step 10: Theoretical evaluation (tau 0.3)
        (['python', pipeline_attack_script, '--config', config_path, '--tau', '0.3'],
         'Step 10: Running pipeline-attack-5.py (Theoretical, tau=0.3)'),
        
        # Step 11: Theoretical evaluation (tau 0.5)
        (['python', pipeline_attack_script, '--config', config_path, '--tau', '0.5'],
         'Step 11: Running pipeline-attack-5.py (Theoretical, tau=0.5)'),
        
        # Step 12: Theoretical evaluation (tau 0.7)
        (['python', pipeline_attack_script, '--config', config_path, '--tau', '0.7'],
         'Step 12: Running pipeline-attack-5.py (Theoretical, tau=0.7)'),
        
        # Step 12.1: Bootstrap without filtering
        (['python', os.path.join(script_dir, 'mia', 'bootstrap_metrics.py'), '--config', config_path, '--n-bootstrap', '100'],
         'Step 12.1: Running bootstrap_metrics.py (without filtering)'),
        
        # Step 12.2: Bootstrap with filtering
        (['python', os.path.join(script_dir, 'mia', 'bootstrap_metrics.py'), '--config', config_path, '--n-bootstrap', '100', '--filter-groundtruth'],
         'Step 12.2: Running bootstrap_metrics.py (with filtering)'),
        
        # Step 13: Compare experimental and theoretical
        (['python', os.path.join(script_dir, 'mia', 'compare_exp_theory.py'), '--config', config_path],
         'Step 13: Running mia/compare_exp_theory.py (Comparison theo/exp)'),
    ]
    
    # If plotting_only is True, only run steps 9-13 (indices 8-14)
    if plotting_only:
        print(f"\n{'='*60}")
        print("PLOTTING ONLY MODE: Running steps 9-13 only")
        print(f"{'='*60}")
        # commands = commands[12:15]  # Steps 9-13 (0-indexed: 8-12, no bootstrap steps)
        # commands = commands[8:15]  # Steps 9-13 (0-indexed: 8-14, includes bootstrap steps)
        commands = [commands[8], commands[12], commands[13], commands[14]]  # Steps 9-13 (0-indexed: 8-12, no bootstrap steps)
    
    for cmd, description in commands:
        if not run_command(cmd, description, script_dir):
            print(f"\n❌ Pipeline failed at: {description}")
            return False
    
    print(f"\n✅ Pipeline completed successfully for config: {config_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run pipeline for all generation files in experimental-recall-all directory'
    )
    parser.add_argument(
        '--template-config',
        type=str,
        default=(REPO_ROOT + '/src/evaluation/pipeline/experimental/config-1B-10-1.0-3_updated.yaml',
        help='Path to template config file'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=(REPO_ROOT + '/outputs/pii_leakage/experimental-recall-all'),
        help='Directory containing generation files'
    )
    parser.add_argument(
        '--config-output-dir',
        type=str,
        default=None,
        help='Directory to save generated config files (default: same as script directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only show what would be run, without executing'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip configs that already have output directories (assumes already processed)'
    )
    parser.add_argument(
        '--plotting-only',
        action='store_true',
        help='Only run plotting steps (steps 9-13: evaluate_scores, theoretical evaluations, and comparison)'
    )
    
    args = parser.parse_args()
    
    # Print mode information
    if args.plotting_only:
        print(f"\n{'='*80}")
        print("PLOTTING ONLY MODE ENABLED")
        print("Will only run steps 9-13 (evaluate_scores, theoretical evaluations, comparison)")
        print(f"{'='*80}\n")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set config output directory
    if args.config_output_dir is None:
        config_output_dir = script_dir
    else:
        config_output_dir = args.config_output_dir
        os.makedirs(config_output_dir, exist_ok=True)
    
    # Load template config
    print(f"Loading template config from: {args.template_config}")
    template_config = load_template_config(args.template_config)
    
    # List all files in input directory
    print(f"\nScanning directory: {args.input_dir}")
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"ERROR: Directory does not exist: {args.input_dir}")
        return 1
    
    generation_files = sorted([f.name for f in input_path.iterdir() 
                              if f.is_file() and f.name.startswith('generation_False_all_') and f.name.endswith('.csv')])
                            #   if f.is_file() and f.name.startswith('generation_False_all_') and f.name.endswith('.parquet')]) #  and '10000000' in f.name # and "3" in f.name
    
    print(f"Found {len(generation_files)} generation files")
    
    # Process each file
    processed_configs = []
    failed_configs = []
    
    for filename in generation_files:
        print(f"\n{'='*80}")
        print(f"Processing: {filename}")
        print(f"{'='*80}")
        
        # Parse parameters from filename
        params = parse_filename(filename)
        if params is None:
            print(f"⚠️  Skipping {filename}: doesn't match expected pattern")
            continue
        
        print(f"Extracted parameters:")
        print(f"  Model: {params['model']}")
        print(f"  Dataset size: {params['dataset_size']}")
        print(f"  PII rate: {params['pii_rate']}")
        print(f"  N epochs: {params['n_epochs']}")
        print(f"  Budget: {params['budget']}")
        
        # Create config filename
        config_filename = f"config-{params['model']}-{params['dataset_size']}-{params['pii_rate']}-{params['n_epochs']}.yaml"
        config_path = os.path.join(config_output_dir, config_filename)
        
        # Check if we should skip existing
        if args.skip_existing:
            # Check if output directory exists (rough heuristic)
            try:
                # Import config_helper from the same directory
                import sys
                sys.path.insert(0, script_dir)
                from config_helper import format_path
                
                output_dir_template = template_config['output_dir']
                test_config = template_config.copy()
                test_config['filters'] = {**template_config['filters'], **params}
                test_config['inputs'] = {**template_config['inputs'], 'budget': params['budget']}
                output_dir = format_path(test_config, output_dir_template)
                if os.path.exists(output_dir) and os.listdir(output_dir):
                    print(f"⏭️  Skipping {filename}: output directory already exists: {output_dir}")
                    continue
            except Exception as e:
                print(f"⚠️  Could not check output directory, proceeding anyway: {e}")
        
        # Create config file
        print(f"\nCreating config file: {config_path}")
        if not args.dry_run:
            create_config_from_template(template_config, params, config_path)
            print(f"✅ Config file created")
        else:
            print(f"[DRY RUN] Would create config file: {config_path}")
        
        # Run pipeline
        if not args.dry_run:
            success = run_pipeline_for_config(config_path, script_dir, plotting_only=args.plotting_only)
            if success:
                processed_configs.append((filename, config_path))
            else:
                failed_configs.append((filename, config_path))
        else:
            mode_str = "plotting steps (9-13)" if args.plotting_only else "full pipeline"
            print(f"[DRY RUN] Would run {mode_str} for: {config_path}")
            processed_configs.append((filename, config_path))
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if args.plotting_only:
        print("Mode: PLOTTING ONLY (steps 9-13)")
    print(f"Total files processed: {len(processed_configs)}")
    print(f"Successful: {len(processed_configs) - len(failed_configs)}")
    print(f"Failed: {len(failed_configs)}")
    
    if failed_configs:
        print(f"\nFailed configs:")
        for filename, config_path in failed_configs:
            print(f"  - {filename} ({config_path})")
    
    return 0 if len(failed_configs) == 0 else 1


if __name__ == '__main__':
    exit(main())
