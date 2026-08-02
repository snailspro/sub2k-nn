

import numpy as np
from nn_sim import QuantizedNN


def confusion_matrix(nn, X, y, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for xi, yi in zip(X, y):
        pred, _ = nn.predict(xi)
        cm[yi][pred] += 1
    return cm


def print_confusion(cm):
    print("       " + " ".join(f"{i:4d}" for i in range(cm.shape[1])) + "   <- previsto")
    for i, row in enumerate(cm):
        print(f"real {i:2d}: " + " ".join(f"{v:4d}" for v in row))


def check_overflow_bounds(W1_q, b1_q, W2_q, b2_q, input_max=16, input_size=64, hidden_size=16):

    worst_acc1 = input_max * 7 * input_size + np.abs(b1_q).max()
    worst_hidden = worst_acc1  
    worst_acc2 = worst_hidden * 7 * hidden_size + np.abs(b2_q).max()

    int32_max = 2**31 - 1
    print(f"[overflow] pior caso teórico do acumulador da camada 1: {worst_acc1:,}")
    print(f"[overflow] pior caso teórico do acumulador da camada 2: {worst_acc2:,}")
    print(f"[overflow] limite do int32_t: {int32_max:,}")
    margem = int32_max / worst_acc2
    print(f"[overflow] margem de segurança: {margem:.1f}x abaixo do limite de overflow")
    assert worst_acc2 < int32_max, "ALERTA: pior caso teórico estoura int32!"


def main():
    data = np.load("pc/nn_table.npz")
    X_test, y_test = data["X_test"], data["y_test"]

    nn = QuantizedNN("pc/nn_table.npz")

    cm = confusion_matrix(nn, X_test, y_test)
    print("=== Matriz de confusão (conjunto de teste, 360 amostras) ===")
    print_confusion(cm)

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\n=== Acurácia por dígito ===")
    for digit, acc in enumerate(per_class_acc):
        print(f"  dígito {digit}: {acc:.1%}")

    overall_acc = cm.diagonal().sum() / cm.sum()
    print(f"\nAcurácia geral: {overall_acc:.1%}")

    print("\n=== Maiores confusões (fora da diagonal) ===")
    confusions = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], i, j))
    confusions.sort(reverse=True)
    for count, real, pred in confusions[:5]:
        print(f"  {count}x: dígito {real} confundido com {pred}")

    print()
    check_overflow_bounds(data["W1_q"], data["b1_q"], data["W2_q"], data["b2_q"])


if __name__ == "__main__":
    main()
