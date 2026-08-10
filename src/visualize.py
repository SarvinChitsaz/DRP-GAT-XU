import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_model_comparison(model_names, r2_scores, colors, save_path):
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, r2_scores, color=colors)
    plt.ylabel("R2 Score")
    plt.title("Model Comparison")
    for i, v in enumerate(r2_scores):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gene_importance(importance_dfs, titles, save_path):
    fig, axes = plt.subplots(1, len(importance_dfs), figsize=(6 * len(importance_dfs), 5))
    for i, (df, title) in enumerate(zip(importance_dfs, titles)):
        axes[i].barh(df["gene"], df["importance"], color="steelblue")
        axes[i].invert_yaxis()
        axes[i].set_title(title, fontsize=10)
        axes[i].set_xlabel("Gradient-based Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_dashboard_validation(validation_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})

    plot_df = validation_df.copy()
    plot_df["status"] = plot_df["actually_sensitive"].map({True: "Correct", False: "Incorrect", None: "No data"})
    colors_map = {"Correct": "seagreen", "Incorrect": "indianred", "No data": "lightgray"}
    bar_colors = plot_df["status"].map(colors_map)

    axes[0].barh(plot_df["patient"] + " (" + plot_df["cancer_type"].str[:15] + ")", plot_df["predicted_probability"], color=bar_colors)
    axes[0].set_xlabel("Predicted Probability of Sensitivity")
    axes[0].set_title("Top-1 Drug Recommendation per Patient")
    axes[0].invert_yaxis()

    legend_elements = [
        Patch(facecolor="seagreen", label="Correct"),
        Patch(facecolor="indianred", label="Incorrect"),
        Patch(facecolor="lightgray", label="No ground truth available"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=8)

    valid_results = validation_df[validation_df["actually_sensitive"].notna()].copy()
    valid_results["actually_sensitive"] = valid_results["actually_sensitive"].astype(bool)
    correct_count = (valid_results["actually_sensitive"] == True).sum()
    incorrect_count = (valid_results["actually_sensitive"] == False).sum()

    axes[1].pie(
        [correct_count, incorrect_count],
        labels=[f"Correct\n({correct_count})", f"Incorrect\n({incorrect_count})"],
        colors=["seagreen", "indianred"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.4},
    )
    axes[1].set_title(f"Validated Hit Rate\n(n={len(valid_results)} patients)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
