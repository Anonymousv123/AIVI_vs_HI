import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load the manuscript data
df = pd.read_csv("emotion_effect_sizes_manuscript.csv")

# 3. Sort strictly by absolute Cohen's h (Largest effect at the top)
plot_df = (
    df.assign(abs_h=df["cohens_h"].abs())
      .sort_values("abs_h", ascending=True) # Ascending=True because barh plots from bottom up
      .tail(12)
      .copy()
)

plot_df["label"] = plot_df["Emotion"].str.capitalize()

# 12 Top Emotion Differences Between AIVI and HI Directed Comments
colors = ["#FF6B6B" if x > 0 else "#4ECDC4" for x in plot_df["Difference_pct_points"]]

# figure size
fig, ax = plt.subplots(figsize=(12, 10))

bars = ax.barh(
    plot_df["label"],
    plot_df["Difference_pct_points"],
    color=colors,
    edgecolor='black',
    linewidth=0.5
)

# Add zero line
ax.axvline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.8)

# Visual expllegened for direction (Text above the axis)
ax.text(3, len(plot_df), "AIVI > HI", color="#FF6B6B", fontsize=16, fontweight='bold', ha='center', va='bottom')
ax.text(-3, len(plot_df), "HI > AIVI", color="#4ECDC4", fontsize=16, fontweight='bold', ha='center', va='bottom')

# Annotate with Cohen's h
for bar, h in zip(bars, plot_df["cohens_h"]):
    x = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    txt = f"h={h:+.3f}"
    # Increased padding to 1.2/-1.2 to avoid crowding
    pad = 1.2 if x >= 0 else -1.2
    ax.text(x + pad, y, txt, va="center", ha="left" if x >= 0 else "right", fontsize=15, fontweight='bold')

# title
ax.set_title("Top 12 Emotion Differences Between AIVI- and HI-Directed Comments", 
             fontsize=20, fontweight='bold', pad=45)

# X-label and Ticks
ax.set_xlabel("Difference in Percentage Points (%AIVI - %HI)", fontsize=18, labelpad=15)
ax.tick_params(axis='both', labelsize=16)

# x-limits
ax.set_xlim(-15, 15)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend at bottom
legend_elements = [Patch(facecolor='#FF6B6B', label='AIVI > HI'),
                   Patch(facecolor='#4ECDC4', label='HI > AIVI')]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=16, ncol=2)

# Adjust layout
plt.subplots_adjust(left=0.25, bottom=0.2, top=0.88)
plt.savefig("figure4_emotions_final_refined.pdf", dpi=300, bbox_inches="tight")
plt.show()
