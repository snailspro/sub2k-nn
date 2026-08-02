import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
NUM_MODELS = 3
HIDDEN_UNITS = 16

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

def train_ensemble(X_train, y_train):
    models = []
    print(f"[Ensemble] Treinando {NUM_MODELS} redes neurais independentes de {HIDDEN_UNITS} neurônios...")
    
    for i in range(NUM_MODELS):
        noise = diophantine_qmc_noise(X_train.shape, scale=1.5, seed_offset=i*999)
        X_train_noisy = X_train + noise
        
        X_train_aug = np.vstack((X_train, X_train_noisy))
        y_train_aug = np.hstack((y_train, y_train))
        
        clf = MLPClassifier(
            hidden_layer_sizes=(HIDDEN_UNITS,),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=3000,
            random_state=RANDOM_STATE + i,
        )
        clf.fit(X_train_aug, y_train_aug)
        models.append(clf)
        print(f"  -> Rede {i+1}/{NUM_MODELS} treinada. Acurácia no treino: {clf.score(X_train, y_train)*100:.1f}%")
        
    return models

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
    return acc2

def evaluate_ensemble_quantized(X, y, ensemble_weights):
    correct = 0
    for xi, yi in zip(X, y):
        total_scores = np.zeros(10, dtype=np.int64)
        for w in ensemble_weights:
            W1_q, b1_q, W2_q, b2_q = w
            scores = forward_int(xi, W1_q, b1_q, W2_q, b2_q)
            total_scores += scores
        
        pred = int(np.argmax(total_scores))
        correct += int(pred == yi)
    return correct / len(X)

def export_c_header(ensemble_weights, active_pixels, filename="firmware/sub2k_nn/nn_table.h"):
    input_dim = ensemble_weights[0][0].shape[0]
    hidden_dim = ensemble_weights[0][0].shape[1]
    output_dim = ensemble_weights[0][2].shape[1]

    with open(filename, "w") as f:
        f.write("// Arquivo gerado automaticamente pelo train_nn.py\n")
        f.write("// ENSEMBLE VOTING: 3 Redes Neurais em PROGMEM\n")
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <avr/pgmspace.h>\n\n")

        f.write(f"#define NUM_MODELS {NUM_MODELS}\n")
        f.write(f"#define INPUT_DIM {input_dim}\n")
        f.write(f"#define HIDDEN_DIM {hidden_dim}\n")
        f.write(f"#define OUTPUT_DIM {output_dim}\n\n")

        f.write(f"const uint8_t active_pixels[{len(active_pixels)}] PROGMEM = {{\n")
        f.write("    " + ", ".join(str(p) for p in active_pixels) + "\n};\n\n")

        for m_idx, w in enumerate(ensemble_weights):
            W1_q, b1_q, W2_q, b2_q = w
            
            # Pack W1
            w1_packed = np.zeros((input_dim, hidden_dim // 2), dtype=np.uint8)
            for i in range(input_dim):
                for j in range(0, hidden_dim, 2):
                    val1 = np.clip(W1_q[i, j], -8, 7).astype(np.int8)
                    val2 = np.clip(W1_q[i, j+1], -8, 7).astype(np.int8)
                    w1_packed[i, j//2] = ((val1 & 0x0F) << 4) | (val2 & 0x0F)
            
            f.write(f"const uint8_t W1_{m_idx}[{input_dim}][{hidden_dim // 2}] PROGMEM = {{\n")
            for row in w1_packed:
                f.write("    {" + ", ".join(f"0x{b:02X}" for b in row) + "},\n")
            f.write("};\n\n")
            
            f.write(f"const int32_t B1_{m_idx}[{hidden_dim}] PROGMEM = {{ " + ", ".join(str(int(v)) for v in b1_q) + " };\n\n")
            
            # Pack W2
            w2_packed = np.zeros((hidden_dim, output_dim // 2), dtype=np.uint8)
            for i in range(hidden_dim):
                for j in range(0, output_dim, 2):
                    val1 = np.clip(W2_q[i, j], -8, 7).astype(np.int8)
                    val2 = np.clip(W2_q[i, j+1], -8, 7).astype(np.int8)
                    w2_packed[i, j//2] = ((val1 & 0x0F) << 4) | (val2 & 0x0F)
            
            f.write(f"const uint8_t W2_{m_idx}[{hidden_dim}][{output_dim // 2}] PROGMEM = {{\n")
            for row in w2_packed:
                f.write("    {" + ", ".join(f"0x{b:02X}" for b in row) + "},\n")
            f.write("};\n\n")
            
            f.write(f"const int32_t B2_{m_idx}[{output_dim}] PROGMEM = {{ " + ", ".join(str(int(v)) for v in b2_q) + " };\n\n")

        f.write(f"const uint8_t* const W1_PTRS[{NUM_MODELS}] PROGMEM = {{\n")
        f.write("    " + ", ".join(f"(const uint8_t*)W1_{i}" for i in range(NUM_MODELS)) + "\n};\n\n")
        
        f.write(f"const int32_t* const B1_PTRS[{NUM_MODELS}] PROGMEM = {{\n")
        f.write("    " + ", ".join(f"B1_{i}" for i in range(NUM_MODELS)) + "\n};\n\n")
        
        f.write(f"const uint8_t* const W2_PTRS[{NUM_MODELS}] PROGMEM = {{\n")
        f.write("    " + ", ".join(f"(const uint8_t*)W2_{i}" for i in range(NUM_MODELS)) + "\n};\n\n")
        
        f.write(f"const int32_t* const B2_PTRS[{NUM_MODELS}] PROGMEM = {{\n")
        f.write("    " + ", ".join(f"B2_{i}" for i in range(NUM_MODELS)) + "\n};\n\n")


def main():
    print("Iniciando treinamento e quantizacao do sub2k-nn (ENSEMBLE VOTING)...")
    
    X_train, X_test, y_train, y_test, active_pixels = load_data()
    print(f"[data] treino: {X_train.shape[0]} amostras | teste: {X_test.shape[0]} amostras | entrada: {X_train.shape[1]} pixels (pruned)")
    
    models = train_ensemble(X_train, y_train)
    
    correct_float = 0
    for xi, yi in zip(X_test, y_test):
        scores = np.zeros(10)
        for clf in models:
            scores += clf.predict_proba([xi])[0]
        if np.argmax(scores) == yi:
            correct_float += 1
    acc_float = correct_float / len(X_test)
    print(f"[float] acurácia no teste (Comitê Float, referência): {acc_float*100:.1f}%")
    
    ensemble_weights = []
    for clf in models:
        ensemble_weights.append(quantize_weights(clf.coefs_, clf.intercepts_))
    
    acc_int_train = evaluate_ensemble_quantized(X_train, y_train, ensemble_weights)
    print(f"[int8] acurácia no treino (Comitê Quantizado): {acc_int_train*100:.1f}%")
    
    acc_int_test = evaluate_ensemble_quantized(X_test, y_test, ensemble_weights)
    print(f"[int8] acurácia no teste  (Comitê Quantizado): {acc_int_test*100:.1f}%")
    print(f"[int8] queda de acurácia por causa da quantização: {(acc_float - acc_int_test)*100:.1f} pontos percentuais")
    
    export_c_header(ensemble_weights, active_pixels)
    
    w1_list = [w[0] for w in ensemble_weights]
    b1_list = [w[1] for w in ensemble_weights]
    w2_list = [w[2] for w in ensemble_weights]
    b2_list = [w[3] for w in ensemble_weights]

    np.savez_compressed("pc/nn_table.npz", 
                        W1_list=w1_list, b1_list=b1_list, 
                        W2_list=w2_list, b2_list=b2_list,
                        X_test=X_test, y_test=y_test, active_pixels=active_pixels)
    print("[export] pc/nn_table.npz salvo (pro simulador/testes)")

if __name__ == "__main__":
    main()
