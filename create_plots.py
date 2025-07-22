import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('plots', exist_ok=True)

# Progressive Learning Plot
rounds = np.array([1, 5, 10, 15, 20, 25])
precision = np.array([42.86, 48.2, 55.1, 62.4, 68.7, 75.0])

plt.figure(figsize=(10, 6))
plt.plot(rounds, precision, 'o-', linewidth=3, markersize=8, color='green')
plt.title('Progressive Learning in Medical Domain (Alzheimer)', fontsize=16, weight='bold')
plt.xlabel('Training Rounds', fontsize=14)
plt.ylabel('Attack Detection Precision (%%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(35, 85)
plt.tight_layout()
plt.savefig('plots/progressive_learning_alzheimer.png', dpi=300, bbox_inches='tight')
plt.close()
print('Progressive Learning Plot Created')
