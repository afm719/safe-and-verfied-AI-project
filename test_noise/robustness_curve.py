import matplotlib.pyplot as plt

# --- DATA (Your results) ---
epsilons = [0.0, 0.001, 0.0012, 0.002]
verified_rates = [100.0, 80.0, 60.0, 15.0]  # % Safe

# --- PLOT CONFIGURATION ---
plt.figure(figsize=(8, 6))
plt.style.use('bmh')  # Clean, academic style

# Main plot line
plt.plot(epsilons, verified_rates, marker='o', linestyle='-', linewidth=2.5, color='#2c3e50', label='Verified Accuracy')

# Shaded area (Safety zone)
plt.fill_between(epsilons, verified_rates, alpha=0.2, color='#3498db')

# Reference line (Reliability Threshold)
plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Reliability Threshold (50%)')

# Titles and Labels (IN ENGLISH)
plt.title('Robustness Curve ($L_\infty$)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Perturbation Radius ($\epsilon$)', fontsize=12)
plt.ylabel('Verified Accuracy (%)', fontsize=12)

# Axis adjustments
plt.ylim(0, 105)
plt.xlim(0, 0.0022)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(frameon=True, loc='upper right')

# Data annotations
for x, y in zip(epsilons, verified_rates):
    plt.annotate(f'{int(y)}%', 
                 (x, y), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 fontsize=10, 
                 fontweight='bold')

# Save and Show
plt.tight_layout()
plt.savefig('robustness_curve.png', dpi=300)
plt.show()

print("Graph generated successfully: robustness_curve.png")