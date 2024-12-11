import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

def load_dataset(csv_file):
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            raise ValueError("Dataset is empty.")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

def generate_summary(df):
    return {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_data": df.head(5).to_dict(),
        "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {}
    }

def write_readme(summary, visualizations):
    readme_content = ["# Automated Analysis Report\n"]
    readme_content.append("## Dataset Summary\n")
    readme_content.append(f"- **Shape**: {summary['shape']}\n")
    readme_content.append(f"- **Columns and Types**: {summary['columns']}\n")
    readme_content.append(f"- **Missing Values**: {summary['missing_values']}\n")
    readme_content.append("- **Sample Data**: \n")
    readme_content.append(f"```\n{pd.DataFrame(summary['sample_data']).to_string(index=False)}\n```\n")

    if 'numeric_summary' in summary and summary['numeric_summary']:
        readme_content.append("## Numeric Summary\n")
        readme_content.append(f"```\n{pd.DataFrame(summary['numeric_summary']).to_string()}\n```\n")

    for vis in visualizations:
        readme_content.append(f"![{vis['title']}]({vis['file']})\n")

    with open("README.md", "w") as f:
        f.writelines(readme_content)

def create_correlation_heatmap(df, output_dir):
    if df.select_dtypes(include=["number"]).shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        heatmap_file = os.path.join(output_dir, "correlation_heatmap.png")
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_file)
        plt.close()
        return {"title": "Correlation Heatmap", "file": heatmap_file}
    return None

def create_missing_values_heatmap(df, output_dir):
    if df.isnull().sum().sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        missing_values_file = os.path.join(output_dir, "missing_values_heatmap.png")
        plt.title("Missing Values Heatmap")
        plt.savefig(missing_values_file)
        plt.close()
        return {"title": "Missing Values Heatmap", "file": missing_values_file}
    return None

def create_histograms(df, output_dir):
    visualizations = []
    for column in df.select_dtypes(include=["number"]):
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True, color="blue")
        plt.title(f"{column} Histogram")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        histogram_file = os.path.join(output_dir, f"{column}_histogram.png")
        plt.savefig(histogram_file)
        plt.close()
        visualizations.append({"title": f"{column} Histogram", "file": histogram_file})
    return visualizations

def create_cluster_plot(df, output_dir):
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        kmeans = KMeans(n_clusters=min(3, len(df)), random_state=42)
        filled_df = numeric_df.fillna(numeric_df.mean())
        clusters = kmeans.fit_predict(filled_df)
        plt.figure(figsize=(10, 6))
        plt.scatter(filled_df.iloc[:, 0], filled_df.iloc[:, 1], c=clusters, cmap="viridis")
        plt.xlabel(filled_df.columns[0])
        plt.ylabel(filled_df.columns[1])
        plt.title("Cluster Analysis")
        cluster_file = os.path.join(output_dir, "cluster_plot.png")
        plt.savefig(cluster_file)
        plt.close()
        return {"title": "Cluster Plot", "file": cluster_file}
    return None

def analyze_and_visualize(csv_file):
    output_dir = os.getcwd()

    # Load dataset
    df = load_dataset(csv_file)

    # Generate summary
    summary = generate_summary(df)

    # Create visualizations
    visualizations = []
    vis = create_correlation_heatmap(df, output_dir)
    if vis:
        visualizations.append(vis)

    vis = create_missing_values_heatmap(df, output_dir)
    if vis:
        visualizations.append(vis)

    visualizations.extend(create_histograms(df, output_dir))

    vis = create_cluster_plot(df, output_dir)
    if vis:
        visualizations.append(vis)

    # Write README.md
    write_readme(summary, visualizations)

    print("Analysis complete. Outputs:")
    print("README.md")
    for vis in visualizations:
        print(vis['file'])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        print("Error: Please provide the path to the CSV file as a command-line argument.")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyze_and_visualize(csv_file)
