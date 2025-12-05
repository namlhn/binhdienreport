import pandas as pd
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import unicodedata
from datetime import datetime, timedelta
import re

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# --- DANH SÁCH 13 THƯƠNG NHÂN + MAP KIOT (chuẩn như trong notebook) ---
NAME_TO_KIOT_RAW = {
    "TĂNG CẨM HẰNG": "T1-015",
    "NGUYỄN THỊ NGỌC SƯƠNG": "TS-010",
    "NGUYỄN THỊ BÍCH VÂN": "T2-045",
    "NGUYỄN HOÀNG NAM": "T1-019",
    "DƯƠNG MINH CƯỜNG": "T2-079",
    "ĐẶNG THỊ THÙY DƯƠNG": "NA-040-041",
    "PHẠM THỊ NGỌC THÚY": "NA-033-034",
    "NGUYỄN THỊ KIM LOAN": "NA-021-022",
    "NGUYỄN THÀNH LONG": "B1-091",
    "HÀ QUỐC VINH": "B1-093",
    "NGUYỄN THỊ NƯƠNG": "B2-124",
    "VŨ PHƯỚC CHÂN": "B3-025",
    "NGUYỄN NGỌC SANG": "B3-063",
}
MERCHANTS_13 = list(NAME_TO_KIOT_RAW.keys())


# --- HÀM CHUẨN HÓA TÊN CỘT (theo notebook) ---
def normalize_text(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def standardize_columns(df: pd.DataFrame, mapping: dict | None = None) -> pd.DataFrame:
    new_cols = {col: normalize_text(col) for col in df.columns}
    df = df.rename(columns=new_cols)
    if mapping:
        df = df.rename(columns=mapping)
    return df


# --- HÀM CHUẨN HÓA TÊN THƯƠNG NHÂN (bỏ dấu, in HOA, gom khoảng trắng) ---
def normalize_name(s):
    if not isinstance(s, str):
        return None
    s = s.strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.upper()
    s = re.sub(r"\s+", " ", s)
    return s


def generate_date_range_labels(year: int):
    # Nhãn ngày cho UI theo định dạng dd/mm, từ 1/12 đến 15/12
    dates = []
    start_date = datetime(year, 12, 1)
    for i in range(15):
        current_day = start_date + timedelta(days=i)
        dates.append(current_day.strftime("%d/%m"))
    return dates

def load_and_process_data():
    dist_path = "data/Distributors.xlsx"
    supp_path = "data/Suppliers.xlsx"

    # Đọc file excel
    df_import_raw = pd.read_excel(supp_path)
    df_export_raw = pd.read_excel(dist_path)

    # Chuẩn hóa tên cột
    df_import_std = standardize_columns(df_import_raw)
    df_export_std = standardize_columns(df_export_raw)

    # Áp mapping Kiot + loại bỏ cột
    drop_cols = ["loai_khai_bao", "thoi_gian_thuc_hien", "thuong_pham", "xuat_su"]
    name_to_kiot = {normalize_name(k): v for k, v in NAME_TO_KIOT_RAW.items()}

    def apply_mapping(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        norm = out.get("nguoi_thuc_hien").map(normalize_name)
        out["kiot"] = norm.map(name_to_kiot)
        to_drop = [c for c in drop_cols if c in out.columns]
        if to_drop:
            out = out.drop(columns=to_drop)
        prefer = [c for c in ["stt", "nguoi_thuc_hien", "kiot", "thoi_gian_tao"] if c in out.columns]
        others = [c for c in out.columns if c not in prefer]
        out = out[prefer + others]
        return out

    df_import_mapped = apply_mapping(df_import_std)
    df_export_mapped = apply_mapping(df_export_std)

    # Thêm cột bucket_date theo mốc 16:00
    def parse_datetime_col(df: pd.DataFrame, col: str) -> pd.Series:
        return pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    def add_bucket_date(df: pd.DataFrame, ts_col: str, bucket_col: str = "bucket_date") -> pd.DataFrame:
        out = df.copy()
        ts = parse_datetime_col(out, ts_col)
        out[bucket_col] = (ts - pd.Timedelta(hours=16)).dt.floor('D') + pd.Timedelta(days=1)
        out[bucket_col] = out[bucket_col].dt.date
        return out

    # Chỉ giữ 13 thương nhân chuẩn
    canon_map = {normalize_name(k): k for k in MERCHANTS_13}

    def assign_canonical_merchant(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        norm = out.get("nguoi_thuc_hien").map(normalize_name)
        out["merchant"] = norm.map(canon_map)
        out = out[out["merchant"].notna()]
        return out

    imp = assign_canonical_merchant(add_bucket_date(df_import_mapped, "thoi_gian_tao"))
    exp = assign_canonical_merchant(add_bucket_date(df_export_mapped, "thoi_gian_tao"))

    # Xác định năm
    def pick_year(*frames):
        for f in frames:
            s = pd.to_datetime(f.get("thoi_gian_tao"), dayfirst=True, errors='coerce')
            yr = s.dt.year.dropna().astype(int)
            if len(yr):
                return int(yr.iloc[0])
        return datetime.now().year

    year = pick_year(imp, exp)

    # Khung ngày 1–15/12
    date_range = pd.date_range(start=datetime(year, 12, 1), end=datetime(year, 12, 15), freq='D').date
    idx = pd.MultiIndex.from_product([MERCHANTS_13, date_range], names=["merchant", "date"])
    base = pd.DataFrame(index=idx).reset_index()

    # Tổng theo ngày-bucket
    imp_cnt = (
        imp.groupby(["merchant", "bucket_date"], dropna=False)
        .size().rename("total_import").reset_index()
        .rename(columns={"bucket_date": "date"})
    )
    exp_cnt = (
        exp.groupby(["merchant", "bucket_date"], dropna=False)
        .size().rename("total_sell").reset_index()
        .rename(columns={"bucket_date": "date"})
    )

    df_daily_totals = base.merge(imp_cnt, on=["merchant", "date"], how="left") \
        .merge(exp_cnt, on=["merchant", "date"], how="left")
    df_daily_totals[["total_import", "total_sell"]] = df_daily_totals[["total_import", "total_sell"]].fillna(0).astype(int)

    # Tổng hợp cho từng thương nhân (toàn kỳ 1–15/12)
    per_merchant = df_daily_totals.groupby("merchant")[["total_import", "total_sell"]].sum().reset_index()
    per_merchant["total_points"] = per_merchant["total_import"] + per_merchant["total_sell"]
    per_merchant = per_merchant.sort_values("total_points", ascending=False)

    # Tổng theo ngày (toàn 13 thương)
    per_date = df_daily_totals.groupby("date")[["total_import", "total_sell"]].sum().reset_index()

    # Top sản phẩm (dựa trên dữ liệu chuẩn hóa trước khi drop, và chỉ 13 thương nhân)
    def top_products(df_std: pd.DataFrame) -> dict:
        tmp = assign_canonical_merchant(df_std)
        if "thuong_pham" in tmp.columns:
            return tmp["thuong_pham"].value_counts().head(5).to_dict()
        return {}

    top_products_sell = top_products(df_import_std)
    top_products_buy = top_products(df_export_std)

    results = {
        "year": year,
        "date_labels": generate_date_range_labels(year),
        "df_daily_totals": df_daily_totals,
        "per_merchant": per_merchant,
        "per_date": per_date,
        "top_products": {"sell": top_products_sell, "buy": top_products_buy},
    }
    return results

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/dashboard-data")
async def get_dashboard_data():
    data = load_and_process_data()

    # Summary
    total_sell = int(data["per_date"]["total_sell"].sum())
    total_buy = int(data["per_date"]["total_import"].sum())

    # Merchants list for UI
    merchants = []
    for _, row in data["per_merchant"].iterrows():
        merchants.append({
            "name": row["merchant"],
            "sell": int(row["total_sell"]),
            "buy": int(row["total_import"]),
            "total_points": int(row["total_points"]),
        })

    active_merchants = sum(1 for m in merchants if (m["sell"] + m["buy"]) > 0)

    # Daily totals array for UI
    daily_totals = []
    for _, r in data["per_date"].sort_values("date").iterrows():
        daily_totals.append({
            "date": r["date"].strftime("%d/%m"),
            "total_import": int(r["total_import"]),
            "total_sell": int(r["total_sell"]),
        })

    return JSONResponse(content={
        "date_range": data["date_labels"],
        "summary": {
            "total_sell": total_sell,
            "total_buy": total_buy,
            "active_merchants": active_merchants,
            "total_merchants_list": len(MERCHANTS_13),
        },
        "merchants": merchants,
        "top_products": data["top_products"],
        "daily_totals": daily_totals,
        # Bổ sung chi tiết theo ngày: merchant + kiot + totals + cumulative_points đến ngày chọn
        "daily_merchant": [
            {
                "date": r_date.strftime("%d/%m"),
                "rows": [
                    (lambda day_rows_for_merchant: {
                        "merchant": m,
                        "kiot": NAME_TO_KIOT_RAW.get(m, ""),
                        "total_import": int(day_rows_for_merchant["total_import"].sum()),
                        "total_sell": int(day_rows_for_merchant["total_sell"].sum()),
                        "cumulative_points": int(df_all.loc[(df_all["merchant"] == m) & (df_all["date"] <= r_date), ["total_import", "total_sell"]].sum().sum()),
                    })(df_all.loc[(df_all["merchant"] == m) & (df_all["date"] == r_date)])
                    for m in MERCHANTS_13
                ]
            }
            for r_date, df_all in (
                (d, data["df_daily_totals"]) for d in data["per_date"]["date"].tolist()
            )
        ],
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)