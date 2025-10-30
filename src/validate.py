import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASELINE_ACCURACY = 0.90  # threshold for CI to pass


def validate_model():
    # Load dataset
    data = load_iris(as_frame=True)
    X, y = data.data, data.target

    # Same split as training for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load trained model
    with open("src/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation Accuracy: {acc:.4f}")

    # Enforce baseline
    assert acc >= BASELINE_ACCURACY, (
        f"Model accuracy {acc:.4f} is below baseline {BASELINE_ACCURACY:.2f}"
    )


if __name__ == "__main__":
    validate_model()
