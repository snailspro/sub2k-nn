

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

HIDDEN_UNITS = 16
RANDOM_STATE = 42


def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) 
    y = digits.target.astype(np.int64) 
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def train_float_model(X_train, y_train):
    rng = np.random.default_rng(RANDOM_STATE)
    noise = rng.uniform(-1.5, 1.5, X_train.shape)
    X_train_noisy = X_train + noise
    
    X_train_aug = np.vstack((X_train, X_train_noisy))
    y_train_aug = np.hstack((y_train, y_train))
    
    print(f"[AMM] Aplicando Ruido Tensorial (Augmentation). Dataset dobrou para {X_train_aug.shape[0]} amostras.")

    clf = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_aug, y_train_aug)
    return clf


def quantize_layer(W: np.ndarray, b: np.ndarray, upstream_scale: float = 1.0):
    max_abs = np.abs(W).max()
    scale = 7.0 / max_abs if max_abs > 0 else 1.0
    W_q = np.round(W * scale).astype(np.int8)

    total_scale = upstream_scale * scale
    b_q = np.round(b * total_scale).astype(np.int32)

    return W_q, b_q, scale, total_scale


def forward_int(x: np.ndarray, W1_q, b1_q, s1, W2_q, b2_q):
    x_int = np.round(x).astype(np.int32)  

    
    acc1 = x_int @ W1_q.astype(np.int32) + b1_q  # (16,)
    hidden = np.maximum(acc1, 0)  

    
    acc2 = hidden.astype(np.int64) @ W2_q.astype(np.int64) + b2_q.astype(np.int64)  

    return int(np.argmax(acc2)), acc2


def evaluate_quantized(X, y, W1_q, b1_q, s1, W2_q, b2_q):
    correct = 0
    for xi, yi in zip(X, y):
        pred, _ = forward_int(xi, W1_q, b1_q, s1, W2_q, b2_q)
        correct += int(pred == yi)
    return correct / len(X)


def export_c_header(path, W1_q, b1_q, W2_q, b2_q, input_size, hidden_units, num_classes):
    with open(path, "w", encoding="utf-8") as f:
        f.write("// Arquivo gerado automaticamente por train_nn.py — não editar à mão.\n")
        f.write("// Projeto: sub2k-nn\n")
        f.write("// Autor: Bruno Nunes da Silva (criador do DevSoft JARVIS AI)\n")
        f.write("// Produto: https://devsoft-ai.webnode.page/\n")
        f.write(f"// input={input_size}  hidden={hidden_units}  classes={num_classes}\n")
        f.write("#pragma once\n#include <avr/pgmspace.h>\n\n")
        f.write(f"#define NN_INPUT_SIZE {input_size}\n")
        f.write(f"#define NN_HIDDEN_SIZE {hidden_units}\n")
        f.write(f"#define NN_NUM_CLASSES {num_classes}\n\n")

        f.write(f"const uint8_t NN_W1[NN_INPUT_SIZE][NN_HIDDEN_SIZE / 2] PROGMEM = {{\n")
        for row in W1_q:
            packed_row = []
            for j in range(0, hidden_units, 2):
                w0 = row[j] & 0x0F
                w1 = row[j+1] & 0x0F
                packed = (w0 << 4) | w1
                packed_row.append(str(packed))
            f.write("  { " + ", ".join(packed_row) + " },\n")
        f.write("};\n\n")

        f.write(f"const int32_t NN_B1[NN_HIDDEN_SIZE] PROGMEM = {{\n  ")
        f.write(", ".join(str(int(v)) for v in b1_q))
        f.write("\n};\n\n")

        f.write(f"const uint8_t NN_W2[NN_HIDDEN_SIZE][NN_NUM_CLASSES / 2] PROGMEM = {{\n")
        for row in W2_q:
            packed_row = []
            for k in range(0, num_classes, 2):
                w0 = row[k] & 0x0F
                w1 = row[k+1] & 0x0F
                packed = (w0 << 4) | w1
                packed_row.append(str(packed))
            f.write("  { " + ", ".join(packed_row) + " },\n")
        f.write("};\n\n")

        f.write(f"const int32_t NN_B2[NN_NUM_CLASSES] PROGMEM = {{\n  ")
        f.write(", ".join(str(int(v)) for v in b2_q))
        f.write("\n};\n")

    total_bytes = (W1_q.size // 2) + b1_q.nbytes + (W2_q.size // 2) + b2_q.nbytes
    print(f"[export] header C escrito em {path} ({total_bytes} bytes de tabela, "
          f"{total_bytes/1024:.2f} KB)")


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"[data] treino: {X_train.shape[0]} amostras | teste: {X_test.shape[0]} amostras "
          f"| entrada: {X_train.shape[1]} pixels | classes: 10")

    clf = train_float_model(X_train, y_train)
    float_acc = clf.score(X_test, y_test)
    print(f"[float] acurácia no teste (modelo float, referência): {float_acc:.1%}")

    W1, W2 = clf.coefs_
    b1, b2 = clf.intercepts_

    W1_q, b1_q, s1, _ = quantize_layer(W1, b1, upstream_scale=1.0)
    W2_q, b2_q, s2, total_scale2 = quantize_layer(W2, b2, upstream_scale=s1)

    quant_acc_train = evaluate_quantized(X_train, y_train, W1_q, b1_q, s1, W2_q, b2_q)
    quant_acc_test = evaluate_quantized(X_test, y_test, W1_q, b1_q, s1, W2_q, b2_q)
    print(f"[int8] acurácia no treino (pesos quantizados, aritmética inteira): {quant_acc_train:.1%}")
    print(f"[int8] acurácia no teste  (pesos quantizados, aritmética inteira): {quant_acc_test:.1%}")
    print(f"[int8] queda de acurácia por causa da quantização: {(float_acc - quant_acc_test)*100:.1f} pontos percentuais")

    export_c_header(
        "firmware/sub2k_nn/nn_table.h", W1_q, b1_q, W2_q, b2_q,
        input_size=X_train.shape[1], hidden_units=HIDDEN_UNITS, num_classes=10,
    )

    np.savez(
        "pc/nn_table.npz",
        W1_q=W1_q, b1_q=b1_q, s1=s1,
        W2_q=W2_q, b2_q=b2_q, s2=s2,
        X_test=X_test, y_test=y_test,
    )
    print("[export] pc/nn_table.npz salvo (pro simulador/testes)")


if __name__ == "__main__":
    main()
