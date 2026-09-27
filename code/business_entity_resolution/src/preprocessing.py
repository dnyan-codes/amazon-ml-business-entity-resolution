import pandas as pd
import re
import unicodedata


def normalize_text(text):
    """
    Normalize text while preserving Unicode characters.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Replace ampersand with 'and'
    text = text.replace("&", " and ")

    # Remove punctuation but preserve Unicode letters/numbers
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df):
    """
    Preprocess business entity data.
    """

    df = df.copy()

    df["business_name_normalized"] = (
        df["business_name"]
        .apply(normalize_text)
    )

    df["business_address_normalized"] = (
        df["business_address"]
        .apply(normalize_text)
    )

    df["country_normalized"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df


def load_source(file_path):
    """
    Load and preprocess a source TSV file.
    """

    df = pd.read_csv(
        file_path,
        sep="\t"
    )

    return preprocess_dataframe(df)


if __name__ == "__main__":
    print("Preprocessing module loaded successfully.")