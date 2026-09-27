import csv
import pandas as pd


def prepare_dataframe(df):
    """
    Create compact blocking keys.
    """

    df = df.copy()

    name = (
        df["business_name_normalized"]
        .fillna("")
        .astype(str)
    )

    address = (
        df["business_address_normalized"]
        .fillna("")
        .astype(str)
    )

    country = (
        df["country_normalized"]
        .fillna("")
        .astype(str)
    )

    # Strong name key
    df["name_key"] = (
        country + "|" + name.str[:6]
    )

    # Strong address key
    df["address_key"] = (
        country + "|" + address.str[:8]
    )

    return df


def build_lookup(df):
    """
    Build lookup:
    blocking key -> entity IDs
    """

    lookup = {}

    for key, group in df.groupby("name_key"):
        lookup[key] = group["entity_id"].tolist()

    return lookup


def build_address_lookup(df):
    """
    Build address blocking lookup.
    """

    lookup = {}

    for key, group in df.groupby("address_key"):
        lookup[key] = group["entity_id"].tolist()

    return lookup


def generate_candidate_pairs(
    source1_df,
    source2_df,
    source3_df,
    output_path,
    max_candidates=100
):
    """
    Generate a manageable candidate_pairs.tsv.

    Candidates are obtained from:
    1. Strong business-name block
    2. Strong address block

    At most max_candidates are retained per Source-1 entity.
    """

    print("Preparing blocking keys...")

    s1 = prepare_dataframe(source1_df)
    s2 = prepare_dataframe(source2_df)
    s3 = prepare_dataframe(source3_df)

    print("Building Source 2 lookups...")

    s2_name_lookup = build_lookup(s2)
    s2_address_lookup = build_address_lookup(s2)

    print("Building Source 3 lookups...")

    s3_name_lookup = build_lookup(s3)
    s3_address_lookup = build_address_lookup(s3)

    total = len(s1)

    print(
        f"Processing {total:,} Source-1 entities..."
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file,
            delimiter="\t"
        )

        writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids"
            ]
        )

        for index, row in enumerate(
            s1.itertuples(index=False),
            start=1
        ):

            candidates = []

            # -----------------------------------------
            # Name candidates
            # -----------------------------------------

            candidates.extend(
                s2_name_lookup.get(
                    row.name_key,
                    []
                )
            )

            candidates.extend(
                s3_name_lookup.get(
                    row.name_key,
                    []
                )
            )

            # -----------------------------------------
            # Address candidates
            # -----------------------------------------

            candidates.extend(
                s2_address_lookup.get(
                    row.address_key,
                    []
                )
            )

            candidates.extend(
                s3_address_lookup.get(
                    row.address_key,
                    []
                )
            )

            # Remove duplicates
            candidates = list(
                dict.fromkeys(candidates)
            )

            # Limit candidates
            candidates = candidates[
                :max_candidates
            ]

            writer.writerow(
                [
                    row.entity_id,
                    ",".join(
                        map(str, candidates)
                    )
                ]
            )

            if index % 10000 == 0:
                print(
                    f"Processed {index:,} / {total:,}"
                )

    print()
    print("Candidate generation completed.")
    print(
        f"Rows written: {total:,}"
    )
    print(
        f"Maximum candidates per S1: "
        f"{max_candidates}"
    )
    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    print(
        "Optimized blocking module loaded."
    )