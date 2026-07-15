import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    # -------------------------
    # 0) Settings (edit here)
    # -------------------------
    seed = 7
    np.random.seed(seed)
    output_dir = "outputs_PGD"
    save_outputs = True
    show_plots = False
    review_marker = 0.50

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

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

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
    clean_cm = confusion_matrix(y_true_sub, y_pred_clean_sub, labels=list(range(10)))
    print(clean_cm)

    # -------------------------
    # 6) PGD sweep over dmax (L2)
    # -------------------------
    solver_params = {"eta": eta, "max_iter": max_iter, "eps": 1e-6}

    adv_acc_list = []
    adv_by_dmax = {}
    robustness_rows = []
    confusion_after_by_dmax = {}

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
        attack_success_rate = float(np.mean(y_pred_adv != y_true_sub))
        accuracy_drop = float(acc_clean_sub - acc_adv)

        perturbations = adv_ds.X.tondarray() - X_sub.tondarray()
        perturbation_norms = np.linalg.norm(perturbations.reshape(perturbations.shape[0], -1), axis=1)
        avg_l2_norm = float(np.mean(perturbation_norms))
        max_l2_norm = float(np.max(perturbation_norms))

        adv_cm = confusion_matrix(y_true_sub, y_pred_adv, labels=list(range(10)))
        confusion_after_by_dmax[dmax] = adv_cm

        adv_acc_list.append(acc_adv)
        robustness_rows.append({
            "dmax": dmax,
            "clean_subset_accuracy": round(float(acc_clean_sub), 4),
            "adversarial_accuracy": round(float(acc_adv), 4),
            "accuracy_drop": round(accuracy_drop, 4),
            "attack_success_rate": round(attack_success_rate, 4),
            "avg_l2_perturbation_norm": round(avg_l2_norm, 4),
            "max_l2_perturbation_norm": round(max_l2_norm, 4),
            "attacked_samples": int(n_attack),
            "max_iter": int(max_iter),
            "eta": float(eta),
        })

        print("Subset adversarial accuracy:", round(acc_adv, 4))
        print("Accuracy drop:", round(accuracy_drop, 4))
        print("Attack success rate:", round(attack_success_rate, 4))
        print("Average L2 perturbation norm:", round(avg_l2_norm, 4))

    if save_outputs:
        summary_path = os.path.join(output_dir, "robustness_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(robustness_rows[0].keys()))
            writer.writeheader()
            writer.writerows(robustness_rows)

        clean_cm_path = os.path.join(output_dir, "confusion_matrix_clean.csv")
        np.savetxt(clean_cm_path, clean_cm, delimiter=",", fmt="%d")

        strongest_dmax = dmax_values[-1]
        adv_cm_path = os.path.join(output_dir, "confusion_matrix_strongest_attack.csv")
        np.savetxt(adv_cm_path, confusion_after_by_dmax[strongest_dmax], delimiter=",", fmt="%d")

        fig_cm, axes_cm = plt.subplots(1, 2, figsize=(10, 4))
        for ax_cm, matrix, title in [
            (axes_cm[0], clean_cm, "Clean subset"),
            (axes_cm[1], confusion_after_by_dmax[strongest_dmax], f"PGD-L2 dmax={strongest_dmax}"),
        ]:
            im = ax_cm.imshow(matrix, cmap="Blues")
            ax_cm.set_title(title)
            ax_cm.set_xlabel("Predicted label")
            ax_cm.set_ylabel("True label")
            ax_cm.set_xticks(range(10))
            ax_cm.set_yticks(range(10))
            for row_idx in range(10):
                for col_idx in range(10):
                    value = int(matrix[row_idx, col_idx])
                    if value:
                        ax_cm.text(col_idx, row_idx, value, ha="center", va="center", fontsize=7)
        fig_cm.colorbar(im, ax=axes_cm.ravel().tolist(), shrink=0.8)
        fig_cm.suptitle("Confusion Matrix Before and After Strongest PGD Attack")
        fig_cm.tight_layout()
        confusion_plot_path = os.path.join(output_dir, "confusion_matrices.png")
        fig_cm.savefig(confusion_plot_path, dpi=160)
        plt.close(fig_cm)

    # -------------------------
    # 7) Plot: accuracy vs dmax
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dmax_values, adv_acc_list, marker="o", label="Adversarial accuracy")
    ax.axhline(review_marker, color="red", linestyle="--", label="Review marker")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("dmax (L2 perturbation budget)")
    ax.set_ylabel("Adversarial accuracy on attacked subset")
    ax.set_title("PGD-L2 Robustness Curve: Accuracy vs Attack Budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_outputs:
        fig.savefig(os.path.join(output_dir, "accuracy_vs_dmax.png"), dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)

    # -------------------------
    # 8) ONE combined visualization for all dmax values
    # -------------------------
    X_clean_np = X_sub.tondarray()
    n_show = min(8, n_attack)

    n_rows = 1 + len(dmax_values)   # 1 clean row + one row per dmax
    n_cols = n_show

    fig = plt.figure(figsize=(2 * n_cols, 2 * n_rows))

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
    if save_outputs:
        fig.savefig(os.path.join(output_dir, "adversarial_examples.png"), dpi=160)
    if show_plots:
        plt.show()
    plt.close(fig)

    if save_outputs:
        first_unreliable = None
        for row in robustness_rows:
            if row["adversarial_accuracy"] < review_marker:
                first_unreliable = row
                break

        final_row = robustness_rows[-1]
        report_lines = [
            "CNN Robustness Evaluation under PGD-L2 Adversarial Attacks",
            "",
            f"Dataset: MNIST",
            f"Training samples used: {n_train}",
            f"Test samples used: {n_test}",
            f"Attacked subset size: {n_attack}",
            f"Clean test accuracy: {acc_test:.4f}",
            f"Clean attacked-subset accuracy: {acc_clean_sub:.4f}",
            "",
            "Attack setup:",
            f"- Attack: PGD-L2",
            f"- max_iter: {max_iter}",
            f"- eta: {eta}",
            f"- dmax sweep: {dmax_values}",
            "",
            "Main result:",
            f"- At the strongest attack budget dmax={final_row['dmax']}, adversarial accuracy dropped to {final_row['adversarial_accuracy']:.4f}.",
            f"- Accuracy drop at strongest attack: {final_row['accuracy_drop']:.4f}.",
            f"- Attack success rate at strongest attack: {final_row['attack_success_rate']:.4f}.",
            f"- Average L2 perturbation norm at strongest attack: {final_row['avg_l2_perturbation_norm']:.4f}.",
            "",
            "Interpretation:",
            "Clean accuracy alone is not enough to judge model reliability. The same CNN that performs well on clean MNIST images becomes much less reliable when small adversarial perturbations are introduced.",
        ]

        if first_unreliable is not None:
            robustness_message = (
                f"The model first falls below the {review_marker:.0%} review marker "
                f"at dmax={first_unreliable['dmax']}."
            )
        else:
            robustness_message = (
                f"The model stayed above the {review_marker:.0%} review marker "
                "for all tested dmax values."
            )
        report_lines.append(robustness_message)

        report_lines.extend([
            "",
            "Saved outputs:",
            "- robustness_summary.csv",
            "- confusion_matrix_clean.csv",
            "- confusion_matrix_strongest_attack.csv",
            "- accuracy_vs_dmax.png",
            "- adversarial_examples.png",
            "- confusion_matrices.png",
        ])

        report_path = os.path.join(output_dir, "attack_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        print("\nSaved robustness outputs:")
        print(f"  - {summary_path}")
        print(f"  - {clean_cm_path}")
        print(f"  - {adv_cm_path}")
        print(f"  - {os.path.join(output_dir, 'accuracy_vs_dmax.png')}")
        print(f"  - {os.path.join(output_dir, 'adversarial_examples.png')}")
        print(f"  - {confusion_plot_path}")
        print(f"  - {report_path}")

        print("\nRobustness interpretation:")
        print(robustness_message)


if __name__ == "__main__":
    main()
