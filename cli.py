import argparse
import numpy as np
import joblib
from explainboostedregg.boosted_explainer import BoostedRegressorExplainer

def main():
    parser = argparse.ArgumentParser(description="Explain a saved regressor")
    parser.add_argument("model", help="Path to joblib model")
    parser.add_argument("X_test", help=".npy file of test features")
    parser.add_argument("y_test", help=".npy file of test targets")
    parser.add_argument("--features", help=".txt file with feature names")
    parser.add_argument("--output-dir", default=".", help="Output folder")
    args = parser.parse_args()

    model = joblib.load(args.model)
    X = np.load(args.X_test)
    y = np.load(args.y_test)
    feat_names = None
    if args.features:
        feat_names = open(args.features).read().splitlines()

    expl = BoostedRegressorExplainer(
        model, X, y, feature_names=feat_names, use_shap=True, perm_kwargs={"n_repeats":5}
    )
    fig1 = expl.plot_global(kind="shap", top_n=10)
    fig1.savefig(f"{args.output_dir}/global_shap.png")
    fig2 = expl.plot_global(kind="perm", top_n=10)
    fig2.savefig(f"{args.output_dir}/global_perm.png")
    fig3 = expl.plot_local(X[0], kind="shap", top_n=10)
    fig3.savefig(f"{args.output_dir}/local_shap.png")
    print("Explanations saved in", args.output_dir)

if __name__ == "__main__":
    main()
