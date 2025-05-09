import os, json
import pandas as pd
import mysql.connector

def fetch_table(query):
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE")
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def main():
    # — 1) Pull core game features + lists from steam_game_details
    df_det = fetch_table("""
        SELECT
          app_id,
          short_description,
          detailed_description,
          is_free,
          price_final      AS price,
          price_discount_percent AS discount,
          genres,
          categories,
          tags
        FROM steam_game_details
    """)

    # — 2) Parse out the list-fields
    # adjust these lambdas to match your storage format
    df_det['genres']     = df_det['genres'].apply(lambda s: [g.strip() for g in s.split(',')] if s else [])
    df_det['categories'] = df_det['categories'].apply(lambda s: [c.strip() for c in s.split(',')] if s else [])
    df_det['tags']       = df_det['tags'].apply(lambda s: [t.strip() for t in s.split(',')] if s else [])

    # — 3) (Optionally) pull bayesian_score, price, discounts as before…
    df_rev = fetch_table("SELECT app_id, bayesian_score FROM steam_game_reviews")
    df = df_det.merge(df_rev, on="app_id", how="left")

    # — 4) Persist a “raw” Parquet you’ll consume downstream
    os.makedirs("/app/ml/data", exist_ok=True)
    df.to_parquet("/app/ml/data/raw_game_features.parquet", index=False)
    print("✅ raw_game_features.parquet written")

    # — 5) Build your global Steam-wide tag list
    df_tag_rows = fetch_table("SELECT tags_json FROM steam_game_tags")
    all_tags = set()
    for j in df_tag_rows['tags_json']:
        try:
            all_tags.update(json.loads(j).get('tags', []))
        except json.JSONDecodeError:
            pass

    with open("/app/ml/data/all_steam_tags.txt", "w", encoding="utf-8") as f:
        for tag in sorted(all_tags):
            f.write(tag + "\n")
    print(f"✅ {len(all_tags)} unique tags written to all_steam_tags.txt")

if __name__ == "__main__":
    main()
