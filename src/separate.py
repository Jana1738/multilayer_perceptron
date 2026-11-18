import pandas as pd
import sys
import os

def separate(path):
    try:
        df = pd.read_csv(path, header=None)
        df = df.drop(columns=df.columns[0])
        features = [
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave_points",
            "symmetry",
            "fractal_dimension",
        ]
        columns = ["diagnosis"]
        for suffix in ["mean", "se", "worst"]:
            columns.extend(f"{feat}_{suffix}" for feat in features)
        df.columns = columns

        b_df = df[df["diagnosis"] == "B"]
        m_df = df[df["diagnosis"] == "M"]       

        b_df= b_df.sample(frac=1, random_state=42)
        m_df= m_df.sample(frac=1, random_state=42)

        b_cut = int(len(b_df) * 0.8)
        m_cut = int(len(m_df) * 0.8)

        train_df = pd.concat([b_df.iloc[:b_cut], m_df.iloc[:m_cut]])
        valid_df = pd.concat([b_df.iloc[b_cut:], m_df.iloc[m_cut:]])

        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)

        os.makedirs("datasets", exist_ok=True)

        train_df.to_csv("datasets/data_training.csv", index=False)
        valid_df.to_csv("datasets/data_validation.csv", index=False)
        print("data splited into training and validation")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(-1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: you should give a datasets as param")
        sys.exit(-1)
    separate(sys.argv[1])