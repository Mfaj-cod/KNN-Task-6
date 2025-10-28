import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier



def train_evaluate(x_train, x_test, y_train, y_test):
    models = {
        "KNN Classifier": KNeighborsClassifier(
            n_neighbors=4,
            weights='uniform',
            algorithm='auto',
            leaf_size=30,
            p=2,
            metric='minkowski',
            metric_params=None,
            n_jobs=None
        )
    }

    reports_dir = os.path.join(os.getcwd(), 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        pdf_path = os.path.join(reports_dir, f"{model_name.replace(' ', '_')}_report.pdf")

        with PdfPages(pdf_path) as pdf:
            # page 1 — Model Report
            fig = plt.figure(figsize=(8.27, 11.69))  # A4
            txt = (
                f"Model: {model_name}\n\n"
                f"Parameters: {model.get_params()}\n\n"
                f"Accuracy: {acc:.4f}\n\n"
                f"Classification Report:\n{classification_report(y_test, y_pred)}"
            )

            plt.text(
                0.05, 0.95, txt,
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment='top',
                family='monospace'
            )
            plt.title(f"{model_name} - Performance Report")
            pdf.savefig(fig)
            plt.close(fig)

            # page 2 — Confusion Matrix Heatmap
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"{model_name} - Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            pdf.savefig(fig)
            plt.close(fig)

            # page 3 — Decision Boundary Visualization (PCA projection for any number of features)

            # Reduce to 2D for visualization
            pca = PCA(n_components=2)
            X_train_2d = pca.fit_transform(x_train)
            X_test_2d = pca.transform(x_test)

            fig, ax = plt.subplots(figsize=(6, 5))

            # Create mesh grid for decision boundary
            x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
            y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                                np.arange(y_min, y_max, 0.05))

            # Predict in PCA space
            Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
            Z = Z.reshape(xx.shape)

            plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
            scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, s=25, edgecolor='k', cmap=plt.cm.coolwarm)
            plt.title(f"{model_name} - Decision Boundary (PCA 2D Projection)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")

            # Add legend for multiclass
            handles, labels = scatter.legend_elements()
            ax.legend(handles, labels, title="Classes", loc="upper right")

            pdf.savefig(fig)
            plt.close(fig)

        print(f"Report with visualization saved at: {pdf_path}\n")
