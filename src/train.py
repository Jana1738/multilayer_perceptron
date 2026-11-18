import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt


def compute_accuracy(y_true, y_pred):
    pred_label = np.argmax(y_pred, axis=1)
    true_label = np.argmax(y_true, axis=1)
        
    return np.mean(true_label == pred_label)

def compute_metrics(y_true, y_pred):
    pred_label = np.argmax(y_pred, axis=1)
    true_label = np.argmax(y_true, axis=1)

    tp = np.sum((true_label == 1) & (pred_label == 1))
    fp = np.sum((true_label == 0) & (pred_label == 1))
    fn = np.sum((true_label == 1) & (pred_label == 0))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def plot(models: list[MLP]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    i = 1
    loss_dict = {}
    acc_dict = {}
    for i, model in enumerate(models, start=1):
        loss_dict[f"train_loss_{i}"] = model.cost_history["train_loss"]
        loss_dict[f"val_loss_{i}"] = model.cost_history["validation_loss"]
        acc_dict[f"train_acc_{i}"] = model.cost_history["train_accuracy"]
        acc_dict[f"val_acc_{i}"] = model.cost_history["validation_accuracy"]

    sns.lineplot(data=loss_dict, ax=axes[0], linewidth=2)
    sns.lineplot(data=acc_dict, ax=axes[1], linewidth=2)

    axes[0].set_title("Categorical Cross-entropy loss", fontsize=12, weight="bold")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    
    axes[1].set_title("Accuracy trend", fontsize=12, weight="bold")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    

    plt.tight_layout()
    plt.show()



class DenseLayer:
    def __init__(self, n_in, n_out, activation="relu", weights_init="HeUniform"):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.weights_init = weights_init

        self.W = self.init_weights()
        self.b = np.zeros((1, n_out))

        self.A_prev = None
        self.A = None
        self.Z = None
        self.dW = None
        self.db = None

        #Buffer pour Adam
        self.t = 0
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)

    def init_weights(self):
        if self.weights_init == "HeUniform":
            limit = np.sqrt(6.0 / self.n_in)
            return np.random.uniform(-limit, limit, size=(self.n_in, self.n_out))
        
        elif self.weights_init == "Xavier":
            limit = np.sqrt(6.0 / (self.n_in + self.n_out))
            return np.random.uniform(-limit, limit, size=(self.n_in, self.n_out))
        
        else:
            raise ValueError(f"Weight initialization method unknown : {self.weights_init}")

    def apply_activation(self, Z):
        if self.activation == "relu":
            A = np.maximum(0, Z)
        elif self.activation == "softmax":
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        else:
            raise ValueError(f"unknown activation method: {self.activation}")
        return A

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W) + self.b
        self.A = self.apply_activation(self.Z)
        return self.A

    def backward(self, grad_in, m):
        if self.activation == "softmax":
            dZ = grad_in
        elif self.activation == "relu":
            dZ = grad_in * (self.Z > 0)
        else:
            raise ValueError(f"unknown activation method: {self.activation}")

        self.dW = (1.0 / m) * np.dot(self.A_prev.T, dZ)
        self.db = (1.0 / m) * np.sum(dZ, axis=0, keepdims=True)

        dA_prev = np.dot(dZ, self.W.T)
        return dA_prev
        

    def update(self, lr, opti = "sgd", beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        if opti == "sgd":
            self.W -= lr * self.dW
            self.b -= lr * self.db
            return
        
        if opti == "adam":
            self.t += 1

            self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
            self.v_W = beta2 * self.v_W + (1 - beta2) * (self.dW**2)

            self.m_b = beta1 * self.m_b + (1 - beta1) * self.db
            self.v_b = beta2 * self.v_b + (1 - beta2) * (self.db**2)

            # 2. Correction du biais d'initialisation (t tend vers l'infini -> 1 - beta^t tend vers 1)
            m_W_hat = self.m_W / (1 - beta1**self.t)
            v_W_hat = self.v_W / (1 - beta2**self.t)

            m_b_hat = self.m_b / (1 - beta1**self.t)
            v_b_hat = self.v_b / (1 - beta2**self.t)

            self.W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)


class MLP:
    def __init__(self, layers: list[DenseLayer]):
        self.layers = layers
        self.cost_history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, y_true, y_pred, lr):
        grad = y_pred - y_true
        m_batch = y_true.shape[0]
        for layer in reversed(self.layers):
            grad = layer.backward(grad, m_batch)
            layer.update(lr)


    def save(self, means, stds):

        weights_dict = {}
        for i, layer in enumerate(self.layers):
            weights_dict[f"W{i + 1}"] = layer.W
            weights_dict[f"b{i + 1}"] = layer.b

        np.savez("model.npz",
                 layers_sizes=np.array([layer.n_in for layer in self.layers] + [self.layers[-1].n_out]),
                 activations=np.array([layer.activation for layer in self.layers]),
                 means=means.to_numpy(),
                 stds=stds.to_numpy(),
                 **weights_dict)


    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size, lr, means, stds):
        m = X_train.shape[0]
        m_val = X_val.shape[0]
        print(f"X_train shape: {X_train.shape}")
        print(f"X_valid shape: {X_val.shape}")
        X_train = X_train.to_numpy()

        
        best_loss = np.inf
        patience = 10
        counter = 0
        min_delta = 1e-4
        best_weights = None

        for _ in range(epochs):
            if counter == patience:
                print(f"stopped at epochs: {_} - best_val_loss: {best_loss}")
                break
            indices = np.random.permutation(m)
            X_shuffled =  X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                A = self.forward(X_batch)
                self.backward(y_batch, A, lr)
            
            A_train = self.forward(X_train)
            cost_train = -(1 / m) * np.sum(y_train * np.log(A_train))
            A_val = self.forward(X_val)
            cost_val = -(1 / m_val) * np.sum(y_val * np.log(A_val))
            print(f"epochs: {_ + 1}/{epochs} - loss: {cost_train} - val_loss: {cost_val}")

            self.cost_history["train_loss"].append(cost_train)
            self.cost_history["validation_loss"].append(cost_val)

            self.cost_history["train_accuracy"].append(compute_accuracy(y_train, A_train))
            self.cost_history["validation_accuracy"].append(compute_accuracy(y_val, A_val))

            if best_loss - cost_val > min_delta:
                best_loss = cost_val
                best_weights = [(layer.W.copy(), layer.b.copy()) for layer in self.layers]
                counter = 0
            else:
                counter += 1


        for layer, (W, b) in zip(self.layers, best_weights):
            layer.W = W
            layer.b = b

        final_train_accuracy = compute_accuracy(y_train, self.forward(X_train))
        final_validation_accuracy = compute_accuracy(y_val, self.forward(X_val))

        print("training accuracy:", final_train_accuracy)
        print("validation accuracy:", final_validation_accuracy)
        self.save(means, stds)


    @classmethod
    def load(cls, filepath="model.npz"):
        with np.load(filepath) as data:
            sizes = data["layers_sizes"]
            activations = data["activations"]

            layers= []
            for i in range(len(activations)):
                layer = DenseLayer(
                    n_in=sizes[i],
                    n_out=sizes[i + 1],
                    activation=activations[i]
                )
                layer.W = data[f"W{i + 1}"]
                layer.b = data[f"b{i + 1}"]
                layers.append(layer)
            means = data["means"]
            stds = data["stds"]
            
        return cls(layers), means, stds


def imput_standardize(X: pd.DataFrame):
    means = X.mean()
    X = X.fillna(means)

    stds = X.std()
    X = (X - means) / stds

    return X, means, stds

def train(path):
    try:
        df_train = pd.read_csv(path[0])
        y_train = np.column_stack([(df_train['diagnosis'] == 'B').astype(int),
                                   (df_train['diagnosis'] == 'M').astype(int)])
        X_train, means, stds = imput_standardize(df_train.drop(columns="diagnosis"))

        df_val = pd.read_csv(path[1])
        y_val = np.column_stack([(df_val['diagnosis'] == 'B').astype(int),
                                   (df_val['diagnosis'] == 'M').astype(int)])
        X_val = df_val.drop(columns="diagnosis")
        X_val = X_val.fillna(means)
        X_val = (X_val - means) / stds

        np.random.seed(42)

        hidden_layer1 = DenseLayer(X_train.shape[1], 24)
        hidden_layer2 = DenseLayer(hidden_layer1.n_out, 16)
        output_layer = DenseLayer(hidden_layer2.n_out, 2, activation="softmax", weights_init="Xavier")

        model = MLP([hidden_layer1, hidden_layer2, output_layer])
        model.fit(X_train, y_train, X_val, y_val, 1000, 32, 0.01, means, stds)

        precision, recall, f1_score = compute_metrics(y_val, model.forward(X_val))
        print(f"val_precision: {precision}")
        print(f"val_recall: {recall}")
        print(f"val_f1_score: {f1_score}")

        plot([model])
        
        
        # hidden_layer3 = DenseLayer(X_train.shape[1], 64)
        # hidden_layer4 = DenseLayer(hidden_layer3.n_out, 32)
        # hidden_layer5 = DenseLayer(hidden_layer4.n_out, 24)
        # hidden_layer6 = DenseLayer(hidden_layer5.n_out, 16)
        # output_layer2 = DenseLayer(hidden_layer6.n_out, 2, activation="softmax", weights_init="Xavier")

        # model2 = MLP([hidden_layer3, hidden_layer4, hidden_layer5, hidden_layer6, output_layer2])
        # model2.fit(X_train, y_train, X_val, y_val, 1000, 32, 0.01, means, stds)
        
        # plot([model, model2])


    except Exception as e:
        print(f"Error: {e}")
        sys.exit(-1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error: please give both training and validation dataset")
        sys.exit(-1)
    train(sys.argv[1:])