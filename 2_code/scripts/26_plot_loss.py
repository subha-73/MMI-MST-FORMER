import pandas as pd
import matplotlib.pyplot as plt
import os

# Data extracted from the user's training log output
data = {
    'Epoch': list(range(1, 51)),
    'Train_MAE': [
        0.7460, 0.7054, 0.6466, 0.5949, 0.5441, 0.5546, 0.5295, 0.5107, 0.5013, 0.4798,
        0.4699, 0.4703, 0.4632, 0.4541, 0.4602, 0.4462, 0.4359, 0.4311, 0.4161, 0.4143,
        0.4089, 0.4029, 0.4062, 0.3924, 0.3916, 0.3939, 0.3816, 0.3811, 0.3822, 0.3729,
        0.3773, 0.3663, 0.3687, 0.3740, 0.3649, 0.3666, 0.3600, 0.3579, 0.3563, 0.3559,
        0.3593, 0.3534, 0.3594, 0.3472, 0.3543, 0.3485, 0.3486, 0.3535, 0.3447, 0.3383
    ],
    'Val_MAE': [
        0.7506, 0.7172, 0.6383, 0.6010, 0.6224, 0.5557, 0.5475, 0.5385, 0.5216, 0.5176,
        0.5183, 0.5186, 0.5220, 0.5022, 0.4975, 0.5114, 0.4881, 0.4916, 0.5029, 0.4823,
        0.4876, 0.4736, 0.4853, 0.4643, 0.4792, 0.4739, 0.4758, 0.4776, 0.4724, 0.4702,
        0.4681, 0.4681, 0.4734, 0.4688, 0.4624, 0.4686, 0.4647, 0.4593, 0.4654, 0.4594,
        0.4693, 0.4673, 0.4601, 0.4715, 0.4720, 0.4696, 0.4644, 0.4836, 0.4647, 0.4606
    ]
}
df = pd.DataFrame(data)

# Find the best validation point
best_val_mae = df['Val_MAE'].min()
best_epoch = df['Val_MAE'].idxmin() + 1

# Create the plot
plt.figure(figsize=(10, 6))

# Plot Training MAE
plt.plot(df['Epoch'], df['Train_MAE'], label='Training MAE', color='blue', marker='o', markersize=3, linestyle='-')

# Plot Validation MAE
plt.plot(df['Epoch'], df['Val_MAE'], label='Validation MAE', color='red', marker='x', markersize=3, linestyle='--')

plt.title('MMI-MST-Former: Forecasting Performance Over Epochs (MAE)')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (MAE) in dB')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Highlight the best validation point
plt.scatter(best_epoch, best_val_mae, color='red', s=150, zorder=5, 
            label=f'Best Val MAE: {best_val_mae:.4f} at Epoch {best_epoch}')
plt.annotate(f'Best: {best_val_mae:.4f}', (best_epoch, best_val_mae), 
             textcoords="offset points", xytext=(5, -15), ha='center', color='red', fontweight='bold')

# Ensure y-axis starts closer to the data to zoom in on convergence
plt.ylim(0.3, 0.8) 

# Save the plot
SAVE_PLOT_PATH = 'mae_loss_graph.png'
plt.savefig(SAVE_PLOT_PATH)
plt.close()

print(f"Loss graph saved to {SAVE_PLOT_PATH}")