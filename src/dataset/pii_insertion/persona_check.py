import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from src._repo import REPO_ROOT
def check_persona(path_1, path_2, split_version):
    # check if only different subjects
    # check name redundancy
    df_1 = pd.read_parquet(path_1)
    df_2 = pd.read_parquet(path_2)

    sid_1 = df_1["subject_id"].unique()
    sid_2 = df_2["subject_id"].unique()
    print("Total number of subjects in df_1: ", len(sid_1))
    print("Total number of subjects in df_2: ", len(sid_2))
    
    ok = True
    
    # Create DataFrame to store intersection counts
    intersection_counts = pd.DataFrame(columns=['column_name', 'unique_values_1', 'unique_values_2', 'intersection_count', 'intersection_values'])

    # compute intersection for subject_ids
    intersection = set(sid_1) & set(sid_2)
    if len(intersection) != 0:
        ok = False
    intersection_counts.loc[len(intersection_counts)] = ['subject_id', len(sid_1), len(sid_2), len(intersection), intersection]
    print(f"Intersection: {intersection}")

    # Check names
    names_1 = df_1["name"].unique()
    names_2 = df_2["name"].unique()
    intersection_names = set(names_1) & set(names_2)
    if len(intersection_names) != 0:
        ok = False
    intersection_counts.loc[len(intersection_counts)] = ['name', len(names_1), len(names_2), len(intersection_names), intersection_names]
    print(f"Intersection names: {intersection_names}")
    print(f"Length of intersection names: {len(intersection_names)}")

    # Check physician names
    physician_names_1 = df_1["physician_name"].unique()
    physician_names_2 = df_2["physician_name"].unique()
    intersection_physician_names = set(physician_names_1) & set(physician_names_2)
    if len(intersection_physician_names) != 0:
        ok = False
    intersection_counts.loc[len(intersection_counts)] = ['physician_name', len(physician_names_1), len(physician_names_2), len(intersection_physician_names), intersection_physician_names]
    print(f"Intersection physician names: {intersection_physician_names}")
    print(f"Length of intersection physician names: {len(intersection_physician_names)}")

    # look at other columns
    for col in df_1.columns:
        if col not in ["subject_id", "name", "physician_name"]:
            values_1 = df_1[col].unique()
            values_2 = df_2[col].unique()
            intersection_values = set(values_1) & set(values_2)
            if len(intersection_values) != 0:
                ok = False
                print(f"Intersection values in column {col}: {intersection_values}")
                print(f"Length of intersection values in column {col}: {len(intersection_values)}")
            intersection_counts.loc[len(intersection_counts)] = [col, len(values_1), len(values_2), len(intersection_values), intersection_values]

    print("\nIntersection Counts Summary:")
    print(intersection_counts)
    intersection_counts.to_csv(f"{REPO_ROOT}/outputs/splits/intersection_counts_{split_version}.csv")
    return ok, intersection_counts

# problem with seed initialization
# def check_duplicates(path):
#     df = pd.read_parquet(path)
#     df_duplicates = df.groupby("subject_id").agg({"name": "first", "physician_name": "first"}).reset_index()
#     # df_duplicates = df_duplicates[df_duplicates.duplicated(subset=["name", "physician_name"])]
#     # df_duplicates = df_duplicates[df_duplicates.duplicated(subset=["name", "physician_name"])]
#     # print(df_duplicates)
#     # print(df_duplicates.duplicated(subset=["name", "physician_name"]))
#     df_duplicates = df_duplicates[["name", "physician_name", "subject_id"]]
#     # print(df_duplicates)
#     k = 0
#     for name in tqdm(df_duplicates["name"]):
#         if len(df_duplicates[df_duplicates['name'] == name]) > 1:
#             # print(df_duplicates[df_duplicates['name'] == name])
#             # print("***", name)
#             k += 1
#     print(f"Number of duplicates: {k}")
    
#     # group duplicates by subject_id
#     # print("XX", df_duplicates)


#     # return df_duplicates
#     return None

def plot_name_distribution(path, title):
    df = pd.read_parquet(path)
    df = df.groupby("subject_id").agg({"name": "first", "physician_name": "first"}).reset_index()
    name_counts = df['name'].value_counts()
    name_proportions = df['name'].value_counts(normalize=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot counts
    name_counts.head(10).plot(kind='bar', ax=ax1)
    ax1.set_title(f'Top 10 Names - Count Distribution ({title})')
    ax1.set_xlabel('Name')
    ax1.set_ylabel('Count')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot proportions
    name_proportions.head(10).plot(kind='bar', ax=ax2)
    ax2.set_title(f'Top 10 Names - Proportion Distribution ({title})')
    ax2.set_xlabel('Name')
    ax2.set_ylabel('Proportion')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{REPO_ROOT}/outputs/splits/name_distribution_{title}.png")
    plt.close()

def check_random_names(path):
    df = pd.read_parquet(path)
    print("Reading", path)
    print(os.path.basename(path), df['random_name'].value_counts())
    for name_check in ["John Doe", "Jane Doe", "John Smith"]:
        check = name_check in df['name'].values
        print(name_check, check)
    print(df[df['subject_id'] == '18341076']['name'])
    
    

def plot_name_duplication_distribution(path, title):
    df = pd.read_parquet(path)
    # Count how many times each name appears
    df = df.groupby("subject_id").agg({"name": "first", "physician_name": "first"}).reset_index()
    # empty_first_name = df['name'].map(lambda x: len(x.split()) == 1)
    empty_first_name = df['name'].map(lambda x: x.split()[0].lower() == "mr.")
    print(df[empty_first_name]['name'].value_counts())
    print(empty_first_name.value_counts())
    df[empty_first_name]['name'].value_counts().to_csv(REPO_ROOT + f'/outputs/splits/name_distribution_{title}_empty_first_name.csv')
    # exit()
    name_counts = df['name'].value_counts()
    # Count how many names appear 1 time, 2 times, etc.
    duplication_counts = name_counts.value_counts().sort_index()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot the distribution
    bars = plt.bar(duplication_counts.index, duplication_counts.values)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.title(f'Distribution of Name Duplications ({title})')
    plt.xlabel('Number of Times Name Appears')
    plt.ylabel('Number of Names')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add summary statistics as text
    total_names = len(name_counts)
    unique_names = len(name_counts[name_counts == 1])
    duplicated_names = total_names - unique_names
    stats_text = f'Total unique names: {total_names}\nNames appearing once: {unique_names}\nNames appearing multiple times: {duplicated_names}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{REPO_ROOT}/outputs/splits/name_duplication_distribution_{title}.png")
    plt.close()

if __name__ == "__main__":
    # version = "v2"
    # version_new = 11
    version_new = 12
    # for version in [1, version_new]:

    for suffix in ["", "_10", "_1"]:
        path = f"{REPO_ROOT}/data/processed/splits_personas_v{version_new}/train{suffix}.parquet"
        check_random_names(path)
    exit()

    for version in [version_new]:
        path_1 = f"{REPO_ROOT}/data/processed/splits_personas_v{version}/train.parquet"
        path_2 = f"{REPO_ROOT}/data/processed/splits_personas_v{version}/val.parquet"
        print(check_persona(path_1, path_2, "v" + str(version)))
        
        # Plot name distributions for both paths
        plot_name_distribution(path_1, f"train_v{version}")
        plot_name_distribution(path_2, f"val_v{version}")
        
        # Plot name duplication distributions
        plot_name_duplication_distribution(path_1, f"train_v{version}")
        plot_name_duplication_distribution(path_2, f"val_v{version}")

    # load the intersection counts for the two versions
    intersection_counts_v1 = pd.read_csv(f"{REPO_ROOT}/outputs/splits/intersection_counts_v1.csv")
    intersection_counts_v1 = intersection_counts_v1.set_index('column_name')
    intersection_counts_v2 = pd.read_csv(f"{REPO_ROOT}/outputs/splits/intersection_counts_v{version_new}.csv")
    intersection_counts_v2 = intersection_counts_v2.set_index('column_name')
    # compute  ratio between intersection counts of v1 and v2
    ratio = intersection_counts_v1['intersection_count'] / intersection_counts_v2['intersection_count']
    ratio = ratio.dropna()

    print("--------------------------------")
    print("Increase in number of unique names compared to v1:")
    print(intersection_counts_v1)
    print(intersection_counts_v2)
    print(ratio)

    # Export combined statistics to LaTeX table
    # Create a DataFrame with the required statistics
    stats_df = pd.DataFrame({
        'Field': intersection_counts_v2.index,
        'Unique Rows Train': intersection_counts_v2['unique_values_1'],
        'Unique Rows Val': intersection_counts_v2['unique_values_2'],
        'Number of Overlaps': intersection_counts_v2['intersection_count']
    })

    # Replace underscores with escaped underscores in column names
    stats_df['Field'] = [col.replace('_', ' ').capitalize() for col in stats_df['Field']]

    # Sort by Field in alphabetical order
    stats_df = stats_df.sort_values('Field')

    # Convert DataFrame to LaTeX table with a label and visible borders
    latex_table = stats_df.to_latex(index=False, label='tab:combined_stats', 
                                   caption='Statistics for Train and Validation Sets',
                                   column_format='lrrr')

    # Add necessary LaTeX packages and center the table
    latex_table = '\\usepackage{booktabs}\n\\usepackage{multirow}\n' + latex_table
    latex_table = latex_table.replace('\\begin{tabular}', '\\begin{center}\\begin{tabular}')
    latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\\end{center}')

    # Save the LaTeX table
    output_path = REPO_ROOT + f'/outputs/splits/combined_stats_v{version_new}.tex'
    with open(output_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table written to {output_path}")

    # check with itself
    # print(check_duplicates(path_1))
    # # print("--------------------------------")
    # print(check_duplicates(path_2))