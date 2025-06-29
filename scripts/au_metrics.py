import os
import sys
import pandas as pd

def compute_and_save_means(csv_path, output_path):
    """
    Read CSV at csv_path, compute mean for each column starting with 'AU',
    and save the resulting means to output_path.
    """
    df = pd.read_csv(csv_path)
    au_cols = [col for col in df.columns if col.startswith('AU')]
    if not au_cols:
        print(f"No AU columns found in {csv_path}")
        return

    # Compute statistics
    means = df[au_cols].mean()
    maxs = df[au_cols].max()
    mins = df[au_cols].min()
    stds = df[au_cols].std()

    # Combine into a single DataFrame with index as statistic names
    summary_df = pd.DataFrame(
        {
            'mean': means,
            'max': maxs,
            'min': mins,
            'std': stds,
        }
    ).T
    
    # Transpose so AUs are columns, stats are rows
    summary_df = summary_df[au_cols]

    summary_df.to_csv(output_path)
    print(f"Saved stats for '{os.path.basename(csv_path)}' to '{output_path}'")


def main():
    if len(sys.argv) != 3:
        print("Usage: python au_metrics.py <experiments_path> <experiment_id>")
        sys.exit(1)

    experiments_path = sys.argv[1]
    experiment_id = sys.argv[2]
    openface_dir = os.path.join(experiments_path, experiment_id, "results/openface")

    # Define input/output filenames
    files = [
        ("au_0.csv", "au_0_means.csv"),
        ("au_1.csv", "au_1_means.csv"),
    ]

    for infile, outfile in files:
        input_path = os.path.join(openface_dir, infile)
        output_path = os.path.join(openface_dir, outfile)

        if not os.path.isfile(input_path):
            print(f"Error: Input file '{input_path}' not found. Skipping.")
            print(f"Please check the path: {experiments_path}/{experiment_id}/results/openface/{infile}")
            continue

        compute_and_save_means(input_path, output_path)


if __name__ == '__main__':
    main()