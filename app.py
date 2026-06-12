import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
import warnings
import re
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Expense Analyzer", page_icon="💰", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: #0d1117 !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── All text white by default ── */
p, span, div, label, li { color: #c9d1d9 !important; }
h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] h2 { color: #58a6ff !important; font-size:1.1rem !important; }
section[data-testid="stSidebar"] .stMultiSelect > div {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: all 0.3s !important;
}
[data-testid="metric-container"]:hover {
    border-color: #58a6ff !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(88,166,255,0.15) !important;
}
[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #30363d !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 12px 24px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
}
.stSuccess > div { 
    background: rgba(35,134,54,0.2) !important;
    border: 1px solid #238636 !important;
    color: #3fb950 !important;
    border-radius: 8px !important;
}
.stInfo > div {
    background: rgba(88,166,255,0.1) !important;
    border: 1px solid #388bfd !important;
    color: #58a6ff !important;
    border-radius: 8px !important;
}

/* ── Code block — fix white background ── */
[data-testid="stCode"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
[data-testid="stCode"] pre {
    background: #161b22 !important;
    color: #c9d1d9 !important;
}
[data-testid="stCode"] code {
    background: #161b22 !important;
    color: #79c0ff !important;
}
.stCodeBlock { background: #161b22 !important; }
.stCodeBlock code { color: #79c0ff !important; }

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    margin: 8px 0 !important;
}
[data-testid="stChatMessage"] p { color: #c9d1d9 !important; }
[data-testid="stChatMessage"] strong { color: #ffffff !important; }

[data-testid="stChatInput"] textarea {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 25px !important;
    color: #ffffff !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #8b949e !important; }
[data-testid="stChatInput"] textarea:focus {
    border-color: #58a6ff !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-size: 0.85rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #58a6ff !important;
    color: #58a6ff !important;
    background: rgba(88,166,255,0.1) !important;
}

/* ── Divider ── */
hr { border-color: #30363d !important; }

/* ── Subheaders ── */
.stMarkdown h3 { color: #ffffff !important; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div style='padding:24px 0 16px 0;'>
    <h1 style='font-size:2rem;font-weight:700;color:#ffffff;margin:0;'>
        💰 AI Expense Analyzer
    </h1>
    <p style='color:#8b949e;margin:4px 0 0 0;font-size:0.95rem;'>
        Smart Financial Insights Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load Data ──
@st.cache_data
def load_data(source):
    if isinstance(source, str):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)
    df.rename(columns={'date_time': 'date'}, inplace=True)
    df.drop(columns=['currency', 'tags'], inplace=True, errors='ignore')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['amount'] > 0]
    df.dropna(inplace=True)
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month_name'] = df['date'].dt.strftime('%b')
    category_map = {
        'Cafe':'Food & Dining','Food':'Food & Dining',
        'Public transport':'Transport','Taxi':'Transport',
        'Health':'Health','Leisure':'Leisure & Fun',
        'Gifts':'Gifts','Clothes':'Shopping',
        'Bought for myself':'Shopping','Other':'Other',
        'Loan given':'Loans','University':'Education','Fines':'Other'
    }
    df['category'] = df['category'].map(category_map).fillna('Other')
    return df

try:
    df = load_data("Expenses_clean.csv")
    st.success(f"✅ Loaded {len(df)} transactions successfully!")
except:
    uploaded = st.file_uploader("Upload Expenses_clean.csv", type="csv")
    if uploaded:
        df = load_data(uploaded)
        st.success(f"✅ Loaded {len(df)} transactions!")
    else:
        st.info("Upload your CSV file to begin")
        st.stop()

# ── Sidebar ──
with st.sidebar:
    st.markdown("##  Filters")
    months = sorted(df['month_name'].unique().tolist())
    selected_months = st.multiselect("Months", months, default=months)
    cats = sorted(df['category'].unique().tolist())
    selected_cats = st.multiselect("Categories", cats, default=cats)
    df = df[df['month_name'].isin(selected_months) & df['category'].isin(selected_cats)]
    st.info(f"Showing {len(df)} transactions")
    st.markdown("---")
    st.markdown("** About**")
    st.markdown("Built with Python, Streamlit & Scikit-learn")

# ── Metrics ──
col1, col2, col3, col4 = st.columns(4)
col1.metric(" TOTAL SPENT", f"₹{df['amount'].sum():,.0f}")
col2.metric("TRANSACTIONS", f"{len(df):,}")
monthly_avg = df.groupby('month')['amount'].sum().mean()
col3.metric("AVG / MONTH", f"₹{monthly_avg:,.0f}")
col4.metric("CATEGORIES", df['category'].nunique())

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([" Overview"," ML Models","🔮Forecast","💬 AI Advisor"])

# ── Colors ──
DARK_BG  = '#0d1117'
SURFACE  = '#161b22'
BLUE     = '#58a6ff'
GREEN    = '#3fb950'
ORANGE   = '#f78166'
YELLOW   = '#e3b341'
PURPLE   = '#bc8cff'
RED      = '#f85149'
TEXT     = '#c9d1d9'
SUBTEXT  = '#8b949e'
BORDER   = '#30363d'

def style_chart(fig, ax):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(SUBTEXT)
    ax.yaxis.label.set_color(SUBTEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, alpha=0.2, color=BORDER, linestyle='--')
    return fig, ax

month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
             7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
monthly_df  = df.groupby('month')['amount'].sum().reset_index()
monthly_df.columns = ['month','total']
cat_totals  = df.groupby('category')['amount'].sum().sort_values(ascending=False)
acct_totals = df.groupby('account')['amount'].sum().sort_values(ascending=False)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Monthly Spending Trend")
        fig, ax = plt.subplots(figsize=(7,4))
        fig, ax = style_chart(fig, ax)
        # ── YELLOW line as requested ──
        ax.plot(monthly_df['month'], monthly_df['total'],
                marker='o', color=YELLOW, linewidth=2.5, markersize=8, zorder=5)
        ax.fill_between(monthly_df['month'], monthly_df['total'],
                        alpha=0.2, color=YELLOW)
        ax.set_xticks(monthly_df['month'])
        ax.set_xticklabels([month_map[m] for m in monthly_df['month']],
                           rotation=45,
