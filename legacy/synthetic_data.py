import os, random
import pandas as pd
import matplotlib.pyplot as plt

def generate_candlestick(ax, bullish=True):
    open_price = random.uniform(10, 50)
    close_price = open_price + random.uniform(1, 10) if bullish else open_price - random.uniform(1, 10)
    high_price = max(open_price, close_price) + random.uniform(1, 5)
    low_price = min(open_price, close_price) - random.uniform(1, 5)
    color = "green" if bullish else "red"
    ax.vlines(0, ymin=low_price, ymax=high_price, colors="black")
    ax.vlines(0, ymin=open_price, ymax=close_price, colors=color, linewidth=6)

def generate_synthetic_dataset(out_dir="data/images", csv_path="data/train.csv", num_samples=200):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for i in range(num_samples):
        bullish = random.choice([True, False])
        label = "bullish" if bullish else "bearish"
        file_path = os.path.join(out_dir, f"img_{i}.png")
        fig, ax = plt.subplots(figsize=(1.2, 2.0))
        ax.axis("off")
        generate_candlestick(ax, bullish=bullish)
        plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        records.append({"image_path": file_path, "label": label})
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Synthetic dataset created: {num_samples} samples at {out_dir}")
