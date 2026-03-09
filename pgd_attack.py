import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    # -------------------------
    # 0) Settings (edit here)
    # -------------------------
    seed = 7
    np.random.seed(seed)

    # Dataset sizes (we use "up to" these values, never exceed available)
    n_train = 60000
    n_test = 10000

    # Attack subset
    n_attack = 200                 # how many test images to attack
    attack_only_correct = True     # recommended: attack only correctly-classified samples

    # Model training (as you asked: 3 epochs)
    epochs = 3
    batch_size = 64
    lr = 0.01

    # Attack type (fixed)
    distance = "l2"
    max_iter = 40                  # keep fixed while sweeping dmax
    eta = 0.2                      # step size (fixed)

    # Sweep ONLY ONE hyperparameter of the attack:
    # dmax = perturbation budget (epsilon) for L2
    dmax_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Pixel bounds after normalization
    lb, ub = 0.0, 1.0

    # -------------------------
    # 1) Imports that need SecML + PyTorch
    # -------------------------
    import torch
    from torch import nn, optim
    from secml.data.loader import CDataLoaderMNIST
    from secml.ml.classifiers import CClassifierPyTorch
    from secml.adv.attacks import CAttackEvasionPGDLS

    torch.manual_seed(seed)

    # -------------------------
    # 2) Simple CNN for MNIST
    # -------------------------
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
            self.drop = nn.Dropout2d(p=0.25)
            self.fc1 = nn.Linear(32 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
            x = torch.relu(torch.max_pool2d(self.drop(self.conv2(x)), 2))
            x = x.view(-1, 32 * 4 * 4)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # -------------------------
    # 3) Load MNIST (SecML)
    # -------------------------
    loader = CDataLoaderMNIST()
    tr = loader.load("training")
    ts = loader.load("testing")

    print("Train samples available:", tr.num_samples)
    print("Test samples available:", ts.num_samples)

    # Use at most requested sizes
    n_train = min(n_train, tr.num_samples)
    n_test = min(n_test, ts.num_samples)

    tr = tr[:n_train, :]
    ts = ts[:n_test, :]

    # Normalize to [0,1]
    tr.X /= 255.0
    ts.X /= 255.0

    # -------------------------
    # 4) Train CNN
    # -------------------------
    model = SimpleCNN()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    clf = CClassifierPyTorch(
        model=model,
        loss=loss_fn,
        optimizer=opt,
        epochs=epochs,
        batch_size=batch_size,
        input_shape=(1, 28, 28),
        random_state=seed
    )

    print("\nTraining...")
    clf.fit(tr.X, tr.Y)
    print("Done.")

    # Clean test accuracy (full test set)
    y_true_test = ts.Y.tondarray().astype(int)
    y_pred_test = clf.predict(ts.X, return_decision_function=False).tondarray().astype(int)
    acc_test = accuracy_score(y_true_test, y_pred_test)
    print("Clean test accuracy:", round(acc_test, 4))

    # -------------------------
    # 5) Choose subset to attack
    # -------------------------
    rng = np.random.default_rng(seed)

    if attack_only_correct:
        correct_ids = np.where(y_pred_test == y_true_test)[0]
        if len(correct_ids) < n_attack:
            n_attack = len(correct_ids)
        idx = rng.permutation(correct_ids)[:n_attack].tolist()
    else:
        idx = rng.permutation(ts.num_samples)[:n_attack].tolist()

    X_sub = ts.X[idx, :]
    y_sub = ts.Y[idx]

    y_true_sub = y_sub.tondarray().astype(int)
    y_pred_clean_sub = clf.predict(X_sub, return_decision_function=False).tondarray().astype(int)
    acc_clean_sub = accuracy_score(y_true_sub, y_pred_clean_sub)

    print("\nSubset clean accuracy:", round(acc_clean_sub, 4))
    print("\nConfusion matrix (clean subset):")
    print(confusion_matrix(y_true_sub, y_pred_clean_sub))

    # -------------------------
    # 6) PGD sweep over dmax (L2)
    # -------------------------
    solver_params = {"eta": eta, "max_iter": max_iter, "eps": 1e-6}

    adv_acc_list = []
    adv_by_dmax = {}

    print("\nPGD sweep (L2): varying dmax only")
    print(f"Fixed: distance={distance}, eta={eta}, max_iter={max_iter}")

    for dmax in dmax_values:
        pgd = CAttackEvasionPGDLS(
            classifier=clf,
            double_init_ds=tr,
            distance=distance,
            dmax=dmax,
            solver_params=solver_params,
            y_target=None,
            lb=lb,
            ub=ub
        )

        print(f"\nRunning PGD-L2 with dmax={dmax} ...")
        _, _, adv_ds, _ = pgd.run(X_sub, y_sub)
        adv_by_dmax[dmax] = adv_ds

        # Evaluate the classifier on the adversarial images
        y_pred_adv = clf.predict(adv_ds.X, return_decision_function=False).tondarray().astype(int)
        acc_adv = accuracy_score(y_true_sub, y_pred_adv)
        adv_acc_list.append(acc_adv)

        print("Subset adversarial accuracy:", round(acc_adv, 4))
        print("Accuracy drop:", round(acc_clean_sub - acc_adv, 4))

    # -------------------------
    # 7) Plot: accuracy vs dmax
    # -------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(dmax_values, adv_acc_list, marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("dmax (L2 perturbation budget)")
    plt.ylabel("Adversarial accuracy on attacked subset")
    plt.title("PGD-L2: Effect of increasing dmax (one attack hyperparameter)")
    plt.grid(True)
    plt.show()

    # -------------------------
    # 8) ONE combined visualization for all dmax values
    # -------------------------
    X_clean_np = X_sub.tondarray()
    n_show = min(8, n_attack)

    n_rows = 1 + len(dmax_values)   # 1 clean row + one row per dmax
    n_cols = n_show

    plt.figure(figsize=(2 * n_cols, 2 * n_rows))

    # Row 1: Clean images
    for i in range(n_show):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(X_clean_np[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.title(f"Clean\nT:{y_true_sub[i]}")

    # Rows for each dmax
    for row_idx, dmax in enumerate(dmax_values):
        adv_ds = adv_by_dmax[dmax]
        X_adv_np = adv_ds.X.tondarray()
        y_pred_adv = clf.predict(adv_ds.X, return_decision_function=False).tondarray().astype(int)

        for col_idx in range(n_show):
            plot_idx = (row_idx + 1) * n_cols + col_idx + 1
            plt.subplot(n_rows, n_cols, plot_idx)
            plt.imshow(X_adv_np[col_idx].reshape(28, 28), cmap="gray")
            plt.axis("off")
            plt.title(f"dmax={dmax}\nP:{y_pred_adv[col_idx]}")

    plt.suptitle("PGD-L2 sweep: increasing dmax (top = clean, below = adversarial)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
