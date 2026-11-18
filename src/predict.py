from train import MLP, DenseLayer
import numpy as np
import sys
import pandas as pd


def predict(data, weights_file):
    
    try:
        model, means, stds = MLP.load(weights_file)
        df = pd.read_csv(data)

        X = df.drop(columns=df.columns[0])
        X = (X - means) / stds
        
        y = (df["diagnosis"] == "M").astype(int)
        
        A = model.forward(X)
        p = A[:, 1]
        for i, prob in enumerate(p):
            print(f"{i}: {prob * 100:.2f}%")
        m = len(y)
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        loss = -(1/m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        print("\nfinal loss:", loss)

    
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error: you should give the dataset to predict and the weights file")
        sys.exit (-1)
    
    predict(sys.argv[1], sys.argv[2])