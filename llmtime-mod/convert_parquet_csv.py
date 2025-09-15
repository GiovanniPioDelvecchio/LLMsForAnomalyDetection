import pandas as pd
import argparse

def parquet_to_csv(input_path, output_path):
    # Carica il file parquet
    df = pd.read_parquet(input_path)

    # Stampa l'head del dataframe
    print("Preview del dataset:")
    print(df.head())

    # Salva in CSV
    df.to_csv(output_path, index=True)
    print(f"\n✅ File salvato come CSV in: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert parquet file to CSV")
    parser.add_argument("input", help="input parquet file")
    parser.add_argument("output", help="output parquet file")
    args = parser.parse_args()

    parquet_to_csv(args.input, args.output)
