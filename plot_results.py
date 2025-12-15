import matplotlib.pyplot as plt
import numpy as np

# Sample data based on the provided table image
# Each model has (mean, std_dev) for each lambda setting
data = {
    'DeepIV': {'pehe': [(30.20, 3.97), (29.16, 5.31), (29.91, 4.70)],
               'ate': [(3.09, 1.51), (1.92, 0.90), (2.56, 1.39)]},
    'GANITE': {'pehe': [(29.51, 1.48), (29.73, 2.59), (30.23, 5.30)],
               'ate': [(1.92, 1.08), (1.68, 1.18), (1.83, 0.89)]},
    'TEDVAE': {'pehe': [(29.94, 2.36), (28.19, 3.11), (27.46, 3.90)],
               'ate': [(2.01, 1.22), (1.94, 0.91), (1.99, 1.02)]},
    'Net-Deconf': {'pehe': [(27.86, 1.65), (28.18, 1.35), (28.12, 1.50)],
                   'ate': [(2.52, 2.56), (2.67, 2.41), (3.95, 3.02)]},
    'Ours': {'pehe': [(27.51, 0.98), (27.97, 1.90), (26.86, 1.51)], 'ate': [(1.87, 1.23), (1.52, 0.85), (1.75, 1.67)]}
}

models = ['DeepIV', 'GANITE', 'TEDVAE', 'Net-Deconf', 'Ours']
metrics = ['pehe', 'ate']
lambda_values = ['Lambda=0.0', 'Lambda=0.5', 'Lambda=1.0']
colors = ['skyblue', 'orange', 'green', 'red', 'purple']

# Create bar positions for each lambda setting
bar_positions = np.arange(len(lambda_values))

# Set up the plots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)

# Plot bars for each metric
for metric_idx, metric in enumerate(metrics):
    for model_idx, model in enumerate(models):
        means = [data[model][metric][lambda_idx][0] for lambda_idx in range(len(lambda_values))]
        std_devs = [data[model][metric][lambda_idx][1] for lambda_idx in range(len(lambda_values))]

        # Offset positions for each model
        offset = (model_idx - len(models) / 2) * 0.1 + 0.05
        positions = bar_positions + offset

        # Plot bars
        axes[metric_idx].bar(positions, means, yerr=std_devs, align='center', alpha=0.7,
                             ecolor='black', capsize=10, color=colors[model_idx], width=0.1,
                             label=f'{model} ({metric.upper()})', error_kw=dict(lw=1, capsize=5, capthick=1))

    # Set labels and titles
    axes[metric_idx].set_ylabel(metric.upper())
    axes[metric_idx].set_title(f'{metric.upper()} Performance for Different Lambda Values')
    axes[metric_idx].set_xticks(bar_positions)
    axes[metric_idx].set_xticklabels(lambda_values)
    axes[metric_idx].grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    axes[metric_idx].legend(loc='upper left')

# Set the x-axis label
plt.xlabel('Lambda Values')

# Tight layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

