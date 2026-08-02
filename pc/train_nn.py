import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV

RANDOM_STATE = 42

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

def diophantine_qmc_noise(shape, scale=1.5):
    N, D = shape
    alpha = 0.6180339887498949  
    total_elements = N * D
    n = np.arange(1, total_elements + 1, dtype=np.uint64)
    k = 3
    P = 1000003  
    n_scrambled = (n ** k) % P
    qmc_01 = (n_scrambled * alpha) % 1.0
    noise = (qmc_01 * 2 * scale) - scale
    return noise.reshape(shape)

def train_float_model(X_train, y_train):
    noise = diophantine_qmc_noise(X_train.shape, scale=1.5)
    X_train_noisy = X_train + noise
    
    X_train_aug = np.vstack((X_train, X_train_noisy))
    y_train_aug = np.hstack((y_train, y_train))
    
    print(f"[AMM] Aplicando Ruido Tensorial DiophantineQMC. Dataset dobrou para {X_train_aug.shape[0]} amostras.")

    print("[GridSearch] Iniciando busca exaustiva (Hyperparameter Tuning)...")
    base_clf = MLPClassifier(
        activation="relu",
        solver="adam",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    
    param_grid = {
        'hidden_layer_sizes': [(16,), (24,), (32,)],
        'alpha': [1e-4, 1e-3, 1e-2],
        'learning_rate_init': [0.001, 0.005]
    }
    
    grid = GridSearchCV(base_clf, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train_aug, y_train_aug)
    
    print(f"[GridSearch] Melhor modelo encontrado: {grid.best_params_}")
    print(f"[GridSearch] Acurácia Cross-Validation: {grid.best_score_*100:.1f}%")
    
    return grid.best_estimator_

def quantize_weights(coefs, intercepts):
    W1, W2 = coefs
    b1, b2 = intercepts
    W1_q = np.round(W1 * 8.0).astype(np.int8)
    W2_q = np.round(W2 * 8.0).astype(np.int8)
    return W1_q, b1.astype(np.int32), W2_q, b2.astype(np.int32)

def forward_int(x: np.ndarray, W1_q, b1_q, W2_q, b2_q):
    x_int = np.round(x).astype(np.int32)
    acc1 = x_int @ W1_q.astype(np.int32) + b1_q
    hidden = np.maximum(acc1, 0)
    acc2 = hidden.astype(np.int64) @ W2_q.astype(np.int64) + b2_q.astype(np.int64)
    return int(np.argmax(acc2)), acc2

def evaluate_quantized(X, y, W1_q, b1_q, W2_q, b2_q):
    correct = 0
    for xi, yi in zip(X, y):
        pred, _ = forward_int(xi, W1_q, b1_q, W2_q, b2_q)
        correct += int(pred == yi)
    return correct / len(X)

def export_c_header(W1_q, b1_q, W2_q, b2_q, active_pixels, filename="firmware/sub2k_nn/nn_table.h"):
    input_dim = W1_q.shape[0]
    hidden_dim = W1_q.shape[1]
    output_dim = W2_q.shape[1]

    w1_packed = np.zeros((input_dim, hidden_dim // 2), dtype=np.uint8)
    for i in range(input_dim):
        for j in range(0, hidden_dim, 2):
            val1 = np.clip(W1_q[i, j], -8, 7).astype(np.int8)
            val2 = np.clip(W1_q[i, j+1], -8, 7).astype(np.int8)
            w1_packed[i, j//2] = ((val1 & 0x0F) << 4) | (val2 & 0x0F)

    w2_packed = np.zeros((hidden_dim, output_dim // 2), dtype=np.uint8)
    for i in range(hidden_dim):
        for j in range(0, output_dim, 2):
            val1 = np.clip(W2_q[i, j], -8, 7).astype(np.int8)
            val2 = np.clip(W2_q[i, j+1], -8, 7).astype(np.int8)
            w2_packed[i, j//2] = ((val1 & 0x0F) << 4) | (val2 & 0x0F)

    with open(filename, "w") as f:
        f.write("// Arquivo gerado automaticamente pelo train_nn.py\n")
        f.write("// Pesos quantizados em INT4 com bit-packing (2 pesos por byte)\n")
        f.write("// Pixels Inativos foram podados.\n")
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <avr/pgmspace.h>\n\n")

        f.write(f"#define INPUT_DIM {input_dim}\n")
        f.write(f"#define HIDDEN_DIM {hidden_dim}\n")
        f.write(f"#define OUTPUT_DIM {output_dim}\n\n")

        f.write(f"const uint8_t active_pixels[{len(active_pixels)}] PROGMEM = {{\n")
        f.write("    " + ", ".join(str(p) for p in active_pixels) + "\n};\n\n")

        f.write(f"const uint8_t W1[{input_dim}][{hidden_dim // 2}] PROGMEM = {{\n")
        for row in w1_packed:
            f.write("    {" + ", ".join(f"0x{b:02X}" for b in row) + "},\n")
        f.write("};\n\n")

        f.write(f"const int32_t B1[{hidden_dim}] PROGMEM = {{ " + ", ".join(str(int(v)) for v in b1_q) + " };\n\n")

        f.write(f"const uint8_t W2[{hidden_dim}][{output_dim // 2}] PROGMEM = {{\n")
        for row in w2_packed:
            f.write("    {" + ", ".join(f"0x{b:02X}" for b in row) + "},\n")
        f.write("};\n\n")

        f.write(f"const int32_t B2[{output_dim}] PROGMEM = {{ " + ", ".join(str(int(v)) for v in b2_q) + " };\n")


def main():
    print("Iniciando treinamento e quantizacao do sub2k-nn (INT4 + Pruning + GridSearch)...")
    
    X_train, X_test, y_train, y_test, active_pixels = load_data()
    print(f"[data] treino: {X_train.shape[0]} amostras | teste: {X_test.shape[0]} amostras | entrada: {X_train.shape[1]} pixels (pruned) | classes: 10")
    
    clf = train_float_model(X_train, y_train)
    acc_float = clf.score(X_test, y_test)
    print(f"[float] acurácia no teste (modelo float, referência): {acc_float*100:.1f}%")
    
    W1_q, b1_q, W2_q, b2_q = quantize_weights(clf.coefs_, clf.intercepts_)
    
    acc_int_train = evaluate_quantized(X_train, y_train, W1_q, b1_q, W2_q, b2_q)
    print(f"[int8] acurácia no treino (pesos quantizados, aritmética inteira): {acc_int_train*100:.1f}%")
    
    acc_int_test = evaluate_quantized(X_test, y_test, W1_q, b1_q, W2_q, b2_q)
    print(f"[int8] acurácia no teste  (pesos quantizados, aritmética inteira): {acc_int_test*100:.1f}%")
    
    print(f"[int8] queda de acurácia por causa da quantização: {(acc_float - acc_int_test)*100:.1f} pontos percentuais")
    
    export_c_header(W1_q, b1_q, W2_q, b2_q, active_pixels)
    
    np.savez_compressed("pc/nn_table.npz", 
                        W1_q=W1_q, b1_q=b1_q, 
                        W2_q=W2_q, b2_q=b2_q,
                        X_test=X_test, y_test=y_test, active_pixels=active_pixels)
    print("[export] pc/nn_table.npz salvo (pro simulador/testes)")

if __name__ == "__main__":
    main()
