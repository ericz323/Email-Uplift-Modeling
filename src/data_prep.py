from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_and_prepare(raw_path=None, treatment_arm="Mens E-Mail", test_size=0.25, random_state=42, save_processed=True):
    raw_path = Path(raw_path) if raw_path is not None else DATA_DIR / "raw" / "hillstrom_data.csv"
    raw_df = pd.read_csv(raw_path)

    filtered_df = raw_df[raw_df["segment"].isin([treatment_arm, "No E-Mail"])].copy()
    filtered_df["treatment"] = (filtered_df["segment"] == treatment_arm).astype(int)

    if save_processed:
        processed_path = DATA_DIR / "processed" / "hillstrom_clean.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(processed_path, index=False)

    strat_key = filtered_df["treatment"].astype(str) + "_" + filtered_df["conversion"].astype(str)
    train_df, test_df = train_test_split(filtered_df, test_size=test_size, random_state=random_state, stratify=strat_key)
    X_train = train_df.drop(columns=["segment", "treatment", "visit", "conversion", "spend"])
    X_test = test_df.drop(columns=["segment", "treatment", "visit", "conversion", "spend"])
    y_train = train_df["conversion"]
    y_test = test_df["conversion"]
    t_train = train_df["treatment"]
    t_test = test_df["treatment"]

    history_segment_order = [
        "1) $0-100", "2) $100-200", "3) $200-350", "4) $350-500",
        "5) $500-750", "6) $750-1000", "7) $1000+",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal_encoder",
                OrdinalEncoder(
                    categories=[history_segment_order],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                ["history_segment"]
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["zip_code", "channel"]
            )
        ],
        remainder="passthrough"
    )

    preprocessor.fit(X_train)

    X_train_encoded = preprocessor.transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    encoded_cols = preprocessor.get_feature_names_out()
    X_train_final = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
    X_test_final = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

    return X_train_final, X_test_final, y_train, y_test, t_train, t_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, t_train, t_test = load_and_prepare()

    for df in [X_train, X_test, y_train, y_test, t_train, t_test]:
        print(df.shape)

    print(y_train.mean())
    print(y_test.mean())
    print(t_train.mean())
    print(t_test.mean())