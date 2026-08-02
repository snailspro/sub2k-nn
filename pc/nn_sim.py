import numpy as np

class QuantizedNN:
    def __init__(self, npz_path="pc/nn_table.npz"):
        data = np.load(npz_path)
        self.W1_list = data["W1_list"]
        self.b1_list = data["b1_list"]
        self.W2_list = data["W2_list"]
        self.b2_list = data["b2_list"]
        self.active_pixels = data["active_pixels"]

    def predict(self, x: np.ndarray):
        if x.shape[0] == 64:
            x = x[self.active_pixels]
            
        x_int = np.round(x).astype(np.int32)
        
        total_scores = np.zeros(10, dtype=np.int64)
        
        for i in range(len(self.W1_list)):
            acc1 = x_int @ self.W1_list[i].astype(np.int32) + self.b1_list[i]
            hidden = np.maximum(acc1, 0)
            acc2 = hidden.astype(np.int64) @ self.W2_list[i].astype(np.int64) + self.b2_list[i].astype(np.int64)
            total_scores += acc2

        return int(np.argmax(total_scores)), total_scores
