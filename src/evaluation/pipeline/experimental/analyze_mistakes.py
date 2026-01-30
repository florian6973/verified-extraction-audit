#!/usr/bin/env python3
"""
Analyze mistakes in the MIA classification results.

A mistake is defined as a value with groundtruth='other' that has a score
above the threshold (i.e., incorrectly classified as a member).
"""

import pandas as pd
import argparse
import os
from src.evaluation.pipeline.experimental.mia.name_filter import name_mask


def analyze_mistakes(csv_path, groundtruth='other', threshold=0.71, score_col='score_oof_member_proba', show_value_found=False, max_display=None):
    """
    Find mistakes where groundtruth='other' and score >= threshold.
    
    Args:
        csv_path: Path to the CSV file with scores
        groundtruth: Groundtruth value to filter (default: 'other')
        threshold: Score threshold (default: 0.71)
        score_col: Name of the score column (default: 'score_oof_member_proba')
    
    Returns:
        DataFrame with mistakes
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check required columns
    required_cols = ['value', 'groundtruth', score_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter for groundtruth='other'
    df_other = df[df['groundtruth'] == groundtruth].copy()
    print(f"\nRows with groundtruth='{groundtruth}': {len(df_other)}")
    
    # Filter for mistakes: score >= threshold
    mistakes = df_other[df_other[score_col] >= threshold].copy()
    print(f"Mistakes (score >= {threshold}): {len(mistakes)}")
    
    if len(mistakes) > 0:
        print(f"\n{'='*80}")
        print(f"MISTAKES (groundtruth='{groundtruth}', threshold={threshold})")
        print(f"{'='*80}")
        
        # Sort by score descending
        mistakes = mistakes.sort_values(score_col, ascending=False)
        
        # Display mistakes
        print(f"\nTotal mistakes: {len(mistakes)}")
        print(f"\nMistakes sorted by score (highest first):")
        print("-" * 80)
        
        display_count = 0
        for idx, row in mistakes.iterrows():
            if max_display is not None and display_count >= max_display:
                remaining = len(mistakes) - display_count
                print(f"\n... ({remaining} more mistakes not displayed)")
                break
            print(f"Value: {row['value']}")
            print(f"  Score: {row[score_col]:.6f}")
            if show_value_found:
                if 'value_found' in row and pd.notna(row['value_found']):
                    print(f"  Found: {str(row['value_found'])[:100]}...")
            print()
            display_count += 1
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Total mistakes: {len(mistakes)}")
        print(f"Min score: {mistakes[score_col].min():.6f}")
        print(f"Max score: {mistakes[score_col].max():.6f}")
        print(f"Mean score: {mistakes[score_col].mean():.6f}")
        print(f"Median score: {mistakes[score_col].median():.6f}")
        
        # Show distribution
        print(f"\nScore distribution:")
        print(mistakes[score_col].describe())
        
        # Apply name filter and show filtered mistakes
        print(f"\n{'='*80}")
        print("MISTAKES AFTER APPLYING NAME FILTER")
        print(f"{'='*80}")
        
        name_mask_result = name_mask(mistakes, column='value')
        filtered_mistakes = mistakes[name_mask_result].copy()
        
        print(f"\nTotal mistakes before filter: {len(mistakes)}")
        print(f"Total mistakes after filter: {len(filtered_mistakes)}")
        print(f"Filtered out: {len(mistakes) - len(filtered_mistakes)}")
        
        if len(filtered_mistakes) > 0:
            # Sort by score descending
            filtered_mistakes = filtered_mistakes.sort_values(score_col, ascending=False)
            
            print(f"\nRemaining mistakes sorted by score (highest first):")
            print("-" * 80)
            
            display_count = 0
            for idx, row in filtered_mistakes.iterrows():
                if max_display is not None and display_count >= max_display:
                    remaining = len(filtered_mistakes) - display_count
                    print(f"\n... ({remaining} more mistakes not displayed)")
                    break
                print(f"Value: {row['value']}")
                print(f"  Score: {row[score_col]:.6f}")

                if show_value_found and 'value_found' in row and pd.notna(row['value_found']):
                    print(f"  Found: {str(row['value_found'])[:100]}...")
                print()
                display_count += 1
            
            # Summary statistics for filtered mistakes
            print(f"\n{'='*80}")
            print("FILTERED MISTAKES SUMMARY STATISTICS")
            print(f"{'='*80}")
            print(f"Total mistakes: {len(filtered_mistakes)}")
            print(f"Min score: {filtered_mistakes[score_col].min():.6f}")
            print(f"Max score: {filtered_mistakes[score_col].max():.6f}")
            print(f"Mean score: {filtered_mistakes[score_col].mean():.6f}")
            print(f"Median score: {filtered_mistakes[score_col].median():.6f}")
            
            print(f"\nFiltered score distribution:")
            print(filtered_mistakes[score_col].describe())
        else:
            print("\nNo mistakes remain after applying the name filter.")
        
    else:
        print(f"\nNo mistakes found for groundtruth='{groundtruth}' and threshold={threshold}")
    
    return mistakes


def main():
    parser = argparse.ArgumentParser(
        description='Analyze mistakes in MIA classification results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze mistakes with default parameters
  python analyze_mistakes.py --csv path/to/file.csv
  
  # Custom threshold
  python analyze_mistakes.py --csv path/to/file.csv --threshold 0.75
  
  # Different groundtruth value
  python analyze_mistakes.py --csv path/to/file.csv --groundtruth train --threshold 0.5
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file with scores'
    )
    parser.add_argument(
        '--groundtruth',
        type=str,
        default='other',
        help='Groundtruth value to filter (default: other)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.71,
        help='Score threshold (default: 0.71)'
    )
    parser.add_argument(
        '--score-col',
        type=str,
        default='score_oof_member_proba',
        help='Name of the score column (default: score_oof_member_proba)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional: Path to save mistakes to CSV file'
    )
    parser.add_argument(
        "--show-value-found",
        action='store_true',
        help='Show the value found for each mistake'
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=None,
        help='Maximum number of examples to display (default: display all)'
    )
    
    args = parser.parse_args()
    
    # Analyze mistakes
    mistakes = analyze_mistakes(
        csv_path=args.csv,
        groundtruth=args.groundtruth,
        threshold=args.threshold,
        score_col=args.score_col,
        show_value_found=args.show_value_found,
        max_display=args.max_display
    )
    
    # Save to file if requested
    if args.output and len(mistakes) > 0:
        mistakes.to_csv(args.output, index=False)
        print(f"\nMistakes saved to: {args.output}")


if __name__ == '__main__':
    main()
