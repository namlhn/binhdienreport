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
    "NGUYỄN THỊ BÍCH VÂN": "T2-045",
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

    # Tổng theo ngày-bucket (Global for reuse)
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

    # Top sản phẩm (dựa trên dữ liệu chuẩn hóa trước khi drop, và chỉ 13 thương nhân)
    def top_products(df_std: pd.DataFrame) -> dict:
        tmp = assign_canonical_merchant(df_std)
        if "thuong_pham" in tmp.columns:
            return tmp["thuong_pham"].value_counts().head(5).to_dict()
        return {}

    top_products_sell = top_products(df_export_std) # Export is Sell
    top_products_buy = top_products(df_import_std)  # Import is Buy

    # --- Helper to calculate stats for a date range ---
    def calculate_period_stats(start_date: datetime.date, end_date: datetime.date, label: str):
        # Filter date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D').date
        
        # Base DataFrame for this range
        idx = pd.MultiIndex.from_product([MERCHANTS_13, date_range], names=["merchant", "date"])
        base = pd.DataFrame(index=idx).reset_index()
        
        # Merge counts
        df_range = base.merge(imp_cnt, on=["merchant", "date"], how="left") \
            .merge(exp_cnt, on=["merchant", "date"], how="left")
        df_range[["total_import", "total_sell"]] = df_range[["total_import", "total_sell"]].fillna(0).astype(int)
        
        # Per merchant stats for this range
        per_merchant = df_range.groupby("merchant")[["total_import", "total_sell"]].sum().reset_index()
        per_merchant["total_points"] = per_merchant["total_import"] + per_merchant["total_sell"]
        per_merchant = per_merchant.sort_values("total_points", ascending=False)
        
        # Per date stats for this range
        per_date = df_range.groupby("date")[["total_import", "total_sell"]].sum().reset_index()
        
        # Summary for this range
        summary = {
            "total_sell": int(per_date["total_sell"].sum()),
            "total_buy": int(per_date["total_import"].sum()),
            # Active merchants: anyone with > 0 points in this period
            "active_merchants": int((per_merchant["total_points"] > 0).sum()),
            "total_merchants_list": len(MERCHANTS_13),
        }
        
        # Merchants list
        merchants_list = []
        for _, row in per_merchant.iterrows():
            merchants_list.append({
                "name": row["merchant"],
                "sell": int(row["total_sell"]),
                "buy": int(row["total_import"]),
                "total_points": int(row["total_points"]),
            })
            
        # Daily totals list
        daily_list = []
        for _, r in per_date.sort_values("date").iterrows():
            daily_list.append({
                "date": r["date"].strftime("%d/%m"),
                "total_import": int(r["total_import"]),
                "total_sell": int(r["total_sell"]),
            })
            
        # Daily merchant details
        daily_merchant_details = [
            {
                "date": r_date.strftime("%d/%m"),
                "rows": [
                    (lambda day_rows: {
                        "merchant": m,
                        "kiot": NAME_TO_KIOT_RAW.get(m, ""),
                        "total_import": int(day_rows["total_import"].sum()),
                        "total_sell": int(day_rows["total_sell"].sum()),
                        # Cumulative points up to this date within the period??
                        # Usually "tích điểm" is cumulative from start of program.
                        # However, for split view, it's ambiguous.
                        # Let's calculate cumulative from START OF THIS PERIOD to show progress within period.
                        "cumulative_points": int(df_range.loc[(df_range["merchant"] == m) & (df_range["date"] <= r_date), ["total_import", "total_sell"]].sum().sum()),
                    })(df_range.loc[(df_range["merchant"] == m) & (df_range["date"] == r_date)])
                    for m in MERCHANTS_13
                ]
            }
            for r_date in date_range
        ]

        return {
            "label": label,
            "summary": summary,
            "merchants": merchants_list, # Leaderboard for this period
            "daily_totals": daily_list,
            "date_range_labels": [d.strftime("%d/%m") for d in date_range],
            "daily_merchant": daily_merchant_details
        }

    # Define ranges
    # Week 1: 1/12 -> 8/12
    # Week 2: 9/12 -> 15/12
    d1 = datetime(year, 12, 1).date()
    d2 = datetime(year, 12, 8).date()
    d2_start = datetime(year, 12, 9).date()
    d3 = datetime(year, 12, 15).date()
    
    week1_stats = calculate_period_stats(d1, d2, "Giai đoạn 1 (01/12 - 08/12)")
    week2_stats = calculate_period_stats(d2_start, d3, "Giai đoạn 2 (09/12 - 15/12)")

    # Legacy results (calculate full 1-15 for fallback or "All" view if needed internally, but we focus on split)
    # Actually, we can just return the periods.

    return {
        "periods": {
            "week1": week1_stats,
            "week2": week2_stats
        },
        "top_products": {"sell": top_products_sell, "buy": top_products_buy}, # Top products remain global for simplicity or per period? 
        # Requirement: "Thống kê tích điểm, số lượng mua bán sẽ chia ra 2 tuần"
        # Since products are minor part, keeping global is safer unless asked. 
        # But logically, products might shift. However, data model for top_products in original code was global.
        # Let's keep top_products global for now to minimize risk, as it wasn't explicitly asked to split.
    }

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/dashboard-data")
async def get_dashboard_data():
    data = load_and_process_data()
    return JSONResponse(content=data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)