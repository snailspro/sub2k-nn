import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
NUM_MODELS = 1
HIDDEN_UNITS = 16
SCALE_W = 127.0

def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) 
    y = digits.target.astype(np.int64) 
    
    variances = np.var(X, axis=0)
    active_pixels = np.argsort(variances)[-42:]
    active_pixels = np.sort(active_pixels).astype(np.uint8)
    
    X = X[:, active_pixels]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    return X_train, X_test, y_train, y_test, active_pixels

def diophantine_qmc_noise(shape, scale=1.5, seed_offset=0):
    N, D = shape
    alpha = 0.6180339887498949  
    total_elements = N * D
    n = np.arange(1 + seed_offset, total_elements + 1 + seed_offset, dtype=np.uint64)
    k = 3
    P = 1000003  
    n_scrambled = (n ** k) % P
    qmc_01 = (n_scrambled * alpha) % 1.0
    noise = (qmc_01 * 2 * scale) - scale
    return noise.reshape(shape)

class CustomQATMLP:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros(output_size)
        
    def q_W(self, w):
        return np.clip(np.round(w * SCALE_W), -128, 127) / SCALE_W
        
    def q_b1(self, b):
        return np.round(b * SCALE_W) / SCALE_W

    def q_b2(self, b):
        scale_sq = SCALE_W * SCALE_W
        return np.round(b * scale_sq) / scale_sq
        
    def forward(self, X):
        self.X = X
        self.W1_q = self.q_W(self.W1)
        self.b1_q = self.q_b1(self.b1)
        self.W2_q = self.q_W(self.W2)
        self.b2_q = self.q_b2(self.b2)
        
        self.Z1 = self.X @ self.W1_q + self.b1_q
        self.A1 = np.maximum(self.Z1, 0)
        self.Z2 = self.A1 @ self.W2_q + self.b2_q
        
        exp_scores = np.exp(self.Z2 - np.max(self.Z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs
        
    def backward(self, y, lr=0.005, weight_decay=1e-4):
        m = y.shape[0]
        dZ2 = self.probs.copy()
        dZ2[range(m), y] -= 1
        dZ2 /= m
        
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)
        
        dA1 = dZ2 @ self.W2_q.T
        dZ1 = dA1.copy()
        dZ1[self.Z1 <= 0] = 0
        
        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)
        
        dW2 += weight_decay * self.W2
        dW1 += weight_decay * self.W1
        
        if not hasattr(self, 'm_W1'):
            self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
            self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
            self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
            self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
            self.t = 0
            
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for w, dw, m_w, v_w in [
            (self.W1, dW1, self.m_W1, self.v_W1),
            (self.b1, db1, self.m_b1, self.v_b1),
            (self.W2, dW2, self.m_W2, self.v_W2),
            (self.b2, db2, self.m_b2, self.v_b2)
        ]:
            m_w[:] = beta1 * m_w + (1 - beta1) * dw
            v_w[:] = beta2 * v_w + (1 - beta2) * (dw ** 2)
            m_hat = m_w / (1 - beta1 ** self.t)
            v_hat = v_w / (1 - beta2 ** self.t)
            w -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
    def train(self, X, y, epochs=1500, lr=0.01):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(y, lr)
            
    def predict_proba(self, X):
        return self.forward(X)
        
    def score(self, X, y):
        probs = self.forward(X)
        return np.mean(np.argmax(probs, axis=1) == y)

def quantize_weights(model):
    W1_q = np.clip(np.round(model.W1 * SCALE_W), -128, 127).astype(np.int8)
    W2_q = np.clip(np.round(model.W2 * SCALE_W), -128, 127).astype(np.int8)
    b1_q = np.round(model.b1 * SCALE_W).astype(np.int32)
    b2_q = np.round(model.b2 * (SCALE_W*SCALE_W)).astype(np.int32)
    return W1_q, b1_q, W2_q, b2_q

def forward_int(x: np.ndarray, W1_q, b1_q, W2_q, b2_q):
    x_int = np.round(x).astype(np.int32)
    acc1 = x_int @ W1_q.astype(np.int32) + b1_q
    hidden = np.maximum(acc1, 0)
    acc2 = hidden.astype(np.int64) @ W2_q.astype(np.int64) + b2_q.astype(np.int64)
    return acc2

def export_c_header(W1_q, b1_q, W2_q, b2_q, active_pixels, filename="firmware/sub2k_nn/nn_table.h"):
    input_dim = W1_q.shape[0]
    hidden_dim = W1_q.shape[1]
    output_dim = W2_q.shape[1]

    with open(filename, "w") as f:
        f.write("// Arquivo gerado automaticamente pelo motor manual QAT (INT8 Nativo)\n")
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <avr/pgmspace.h>\n\n")

        f.write(f"#define INPUT_DIM {input_dim}\n")
        f.write(f"#define HIDDEN_DIM {hidden_dim}\n")
        f.write(f"#define OUTPUT_DIM {output_dim}\n\n")

        f.write(f"const uint8_t active_pixels[{len(active_pixels)}] PROGMEM = {{\n")
        f.write("    " + ", ".join(str(p) for p in active_pixels) + "\n};\n\n")

        f.write(f"const int8_t W1[{input_dim}][{hidden_dim}] PROGMEM = {{\n")
        for row in W1_q:
            f.write("    {" + ", ".join(str(int(b)) for b in row) + "},\n")
        f.write("};\n\n")
        
        f.write(f"const int32_t B1[{hidden_dim}] PROGMEM = {{ " + ", ".join(str(int(v)) for v in b1_q) + " };\n\n")
        
        f.write(f"const int8_t W2[{hidden_dim}][{output_dim}] PROGMEM = {{\n")
        for row in W2_q:
            f.write("    {" + ", ".join(str(int(b)) for b in row) + "},\n")
        f.write("};\n\n")
        
        f.write(f"const int32_t B2[{output_dim}] PROGMEM = {{ " + ", ".join(str(int(v)) for v in b2_q) + " };\n\n")


def main():
    print("Iniciando motor matemático puro (Numpy QAT INT8 Nativo)...")
    
    X_train, X_test, y_train, y_test, active_pixels = load_data()
    print(f"[data] treino: {X_train.shape[0]} amostras | teste: {X_test.shape[0]} amostras | entrada: {X_train.shape[1]} pixels (pruned)")
    
    noise = diophantine_qmc_noise(X_train.shape, scale=1.5, seed_offset=999)
    X_train_noisy = X_train + noise
    
    X_train_aug = np.vstack((X_train, X_train_noisy))
    y_train_aug = np.hstack((y_train, y_train))
    
    clf = CustomQATMLP(input_size=X_train.shape[1], hidden_size=HIDDEN_UNITS, output_size=10, seed=RANDOM_STATE)
    clf.train(X_train_aug, y_train_aug, epochs=2500, lr=0.005)
    
    print(f"[float] acurácia Float (QAT) no treino: {clf.score(X_train, y_train)*100:.1f}%")
    print(f"[float] acurácia Float (QAT) no teste: {clf.score(X_test, y_test)*100:.1f}%")
    
    W1_q, b1_q, W2_q, b2_q = quantize_weights(clf)
    
    correct = 0
    for xi, yi in zip(X_test, y_test):
        scores = forward_int(xi, W1_q, b1_q, W2_q, b2_q)
        if np.argmax(scores) == yi: correct += 1
    acc_int_test = correct / len(X_test)
    
    print(f"[int8] acurácia no teste  (Arduino INT8 Nativo): {acc_int_test*100:.1f}%")
    
    export_c_header(W1_q, b1_q, W2_q, b2_q, active_pixels)
    
    np.savez_compressed("pc/nn_table.npz", 
                        W1=W1_q, b1=b1_q, 
                        W2=W2_q, b2=b2_q,
                        X_test=X_test, y_test=y_test, active_pixels=active_pixels)
    print("[export] pc/nn_table.npz salvo (pro simulador/testes)")

if __name__ == "__main__":
    main()
