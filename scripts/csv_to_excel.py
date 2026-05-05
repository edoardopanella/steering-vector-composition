"""Convert results/human_eval_layer{LAYER}.csv into two .xlsx files:

  * `..._full.xlsx`   — all columns including `setting`, original row order.
                       Use this for the final merge with judge scores.
  * `..._blind.xlsx`  — `setting` column dropped, rows shuffled.
                       Use this for the unbiased human rating pass; the human
                       cannot infer the steering setting from column or position.

An `id` column is added before writing so blind ratings can be merged back into
the full table after evaluation. Set RANDOM_SEED to make the shuffle reproducible.
"""

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

LAYER = 17
RANDOM_SEED = 0
IN_PATH = Path(f"results/human_eval_layer{LAYER}.csv")
FULL_PATH = IN_PATH.with_name(f"human_eval_layer{LAYER}_full.xlsx")
BLIND_PATH = IN_PATH.with_name(f"human_eval_layer{LAYER}_blind.xlsx")

COLUMN_WIDTHS = {
    "id": 6,
    "behavior_pair": 22,
    "setting": 10,
    "prompt": 50,
    "completion": 80,
    "rating_b1": 12,
    "rating_b2": 12,
    "notes": 30,
}


def write_xlsx(df: pd.DataFrame, path: Path, sheet: str) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet)
        ws = writer.sheets[sheet]
        ws.freeze_panes = "A2"
        for idx, col in enumerate(df.columns, start=1):
            ws.column_dimensions[get_column_letter(idx)].width = COLUMN_WIDTHS.get(col, 18)
        wrap = Alignment(wrap_text=True, vertical="top")
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
            for cell in row:
                cell.alignment = wrap


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df.insert(0, "id", range(len(df)))

    write_xlsx(df, FULL_PATH, sheet="human_eval_full")
    print(f"Wrote {len(df)} rows to {FULL_PATH}")

    blind = df.drop(columns=["setting"]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    write_xlsx(blind, BLIND_PATH, sheet="human_eval_blind")
    print(f"Wrote {len(blind)} rows to {BLIND_PATH}  (setting hidden, rows shuffled)")
