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
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
}

#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117, #161b22) !important;
    border-right: 1px solid #30363d !important;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stMultiSelect div {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] h2 {
    color: #58a6ff !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161b22, #21262d) !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}
[data-testid="metric-container"]:hover {
    border-color: #58a6ff !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(88,166,255,0.15) !important;
}
[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #30363d !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    border-radius: 0 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 12px 24px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}

.stSuccess {
    background: rgba(35,134,54,0.15) !important;
    border: 1px solid rgba(35,134,54,0.4) !important;
    border-radius: 8px !important;
    color: #3fb950 !important;
}
.stInfo {
    background: rgba(88,166,255,0.1) !important;
    border: 1px solid rgba(88,166,255,0.3) !important;
    border-radius: 8px !important;
    color: #58a6ff !important;
}

h1, h2, h3 { color: white !important; }
.stMarkdown p { color: #c9d1d9 !important; }

[data-testid="stChatMessage"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    margin: 8px 0 !important;
    padding: 12px !important;
}
[data-testid="stChatMessage"] p { color: #c9d1d9 !important; }

[data-testid="stChatInput"] textarea {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 25px !important;
    color: white !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}

.stButton button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton button:hover {
    border-color: #58a6ff !important;
    color: #58a6ff !important;
    background: rgba(88,166,255,0.1) !important;
}

.stCode {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
hr { border-color: #30363d !important; }

[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 2px dashed #30363d !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #58a6ff !important;
}
[data-testid="stFileUploader"] label {
    color: #8b949e !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div style='padding: 24px 0 16px 0;'>
    <h1 style='font-size:2rem; font-weight:700; color:white; margin:0;'>
        💰 AI Expense Analyzer
    </h1>
    <p style='color:#8b949e; margin:4px 0 0 0; font-size:0.95rem;'>
        Smart Financial Insights Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ── Data Loading ──
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
        'Cafe': 'Food & Dining', 'Food': 'Food & Dining',
        'Public transport': 'Transport', 'Taxi': 'Transport',
        'Health': 'Health', 'Leisure': 'Leisure & Fun',
        'Gifts': 'Gifts', 'Clothes': 'Shopping',
        'Bought for myself': 'Shopping', 'Other': 'Other',
        'Loan given': 'Loans', 'University': 'Education', 'Fines': 'Other'
    }
    df['category'] = df['category'].map(category_map).fillna('Other')
    return df

# ── Try auto-load first, fallback to upload ──
try:
    df = load_data("Expenses_clean.csv")
    st.success(f"✅ Loaded {len(df)} transactions successfully!")
except:
    st.markdown("### 📂 Upload Your Expense Data")
    uploaded = st.file_uploader("Upload Expenses_clean.csv", type="csv")
    if uploaded:
        df = load_data(uploaded)
        st.success(f"✅ Loaded {len(df)} transactions!")
    else:
        st.info("👆 Upload your CSV file to begin")
        st.markdown("""
        **What this dashboard includes:**
        - 📊 Monthly trends, category & account analysis
        - 🤖 Random Forest ML model for category prediction
        - 🔮 3-month expense forecast
        - 💬 AI Financial Advisor chatbot
        """)
        st.stop()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🔍 Filters")
    months = sorted(df['month_name'].unique().tolist())
    selected_months = st.multiselect("Months", months, default=months)
    cats = sorted(df['category'].unique().tolist())
    selected_cats = st.multiselect("Categories", cats, default=cats)
    df = df[df['month_name'].isin(selected_months) & df['category'].isin(selected_cats)]
    st.info(f"Showing {len(df)} transactions")
    st.markdown("---")
    st.markdown("**📌 About**")
    st.markdown("Built with Python, Streamlit & Scikit-learn")

# ── Metrics ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("💸 TOTAL SPENT", f"₹{df['amount'].sum():,.0f}")
col2.metric("🧾 TRANSACTIONS", f"{len(df):,}")
monthly_avg = df.groupby('month')['amount'].sum().mean()
col3.metric("📅 AVG / MONTH", f"₹{monthly_avg:,.0f}")
col4.metric("🗂️ CATEGORIES", df['category'].nunique())

st.divider()

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 ML Models", "🔮 Forecast", "💬 AI Advisor"])

DARK_BG = '#0d1117'
SURFACE = '#161b22'
BLUE = '#58a6ff'
GREEN = '#3fb950'
ORANGE = '#f78166'
YELLOW = '#e3b341'
PURPLE = '#bc8cff'
RED = '#f85149'

def style_chart(fig, ax):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(True, alpha=0.2, color='#30363d', linestyle='--')
    return fig, ax

month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
             7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
monthly_df = df.groupby('month')['amount'].sum().reset_index()
monthly_df.columns = ['month', 'total']
cat_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
acct_totals = df.groupby('account')['amount'].sum().sort_values(ascending=False)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Monthly Spending Trend")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig, ax = style_chart(fig, ax)
        ax.plot(monthly_df['month'], monthly_df['total'], marker='o',
                color=BLUE, linewidth=2.5, markersize=7, zorder=5)
        ax.fill_between(monthly_df['month'], monthly_df['total'], alpha=0.15, color=BLUE)
        ax.set_xticks(monthly_df['month'])
        ax.set_xticklabels([month_map[m] for m in monthly_df['month']], rotation=45)
        ax.set_ylabel("Amount (₹)")
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.subheader("Spending by Category")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor(SURFACE)
        colors = [BLUE, GREEN, YELLOW, ORANGE, PURPLE, RED, '#1abc9c', '#e67e22', '#95a5a6']
        top = cat_totals.head(7)
        others = cat_totals.iloc[7:].sum()
        vals = list(top.values) + ([others] if others > 0 else [])
        lbls = list(top.index) + (['Others'] if others > 0 else [])
        wedges, texts, autotexts = ax2.pie(vals, labels=lbls, autopct='%1.1f%%',
                                            colors=colors[:len(vals)], startangle=90,
                                            pctdistance=0.85)
        for t in texts: t.set_color('#c9d1d9'); t.set_fontsize(9)
        for a in autotexts: a.set_color('white'); a.set_fontweight('bold'); a.set_fontsize(8)
        ax2.set_facecolor(SURFACE)
        plt.tight_layout()
        st.pyplot(fig2)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Spending by Account")
        fig3, ax3 = plt.subplots(figsize=(7, 3))
        fig3, ax3 = style_chart(fig3, ax3)
        bar_colors = [BLUE, GREEN, PURPLE, YELLOW, RED]
        bars = ax3.bar(range(len(acct_totals)), acct_totals.values,
                       color=bar_colors[:len(acct_totals)], width=0.5, edgecolor='none')
        ax3.set_xticks(range(len(acct_totals)))
        ax3.set_xticklabels(acct_totals.index, rotation=20)
        ax3.set_ylabel("Amount (₹)")
        for bar, val in zip(bars, acct_totals.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'₹{val:,.0f}', ha='center', color='#8b949e', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig3)

    with c4:
        st.subheader("Spending by Day of Week")
        fig4, ax4 = plt.subplots(figsize=(7, 3))
        fig4, ax4 = style_chart(fig4, ax4)
        dow = df.groupby('day_of_week')['amount'].sum()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        bar_cols = [GREEN if i < 5 else ORANGE for i in dow.index]
        ax4.bar(range(len(dow)), dow.values, color=bar_cols, width=0.6, edgecolor='none')
        ax4.set_xticks(range(len(dow)))
        ax4.set_xticklabels([day_names[i] for i in dow.index])
        ax4.set_ylabel("Amount (₹)")
        plt.tight_layout()
        st.pyplot(fig4)

with tab2:
    st.subheader("🤖 Random Forest Classifier")

    @st.cache_resource
    def train(_df):
        le_cat = LabelEncoder()
        le_acct = LabelEncoder()
        _df = _df.copy()
        _df['category_encoded'] = le_cat.fit_transform(_df['category'])
        _df['account_encoded'] = le_acct.fit_transform(_df['account'])
        X = _df[['amount', 'month', 'day_of_week', 'week', 'account_encoded']]
        y = _df['category_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, target_names=le_cat.classes_, zero_division=0)
        return acc, report

    with st.spinner("⚙️ Training model..."):
        acc, report = train(df)

    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 Accuracy", f"{acc*100:.1f}%")
    m2.metric("🌲 Trees", "100")
    m3.metric("📊 Features", "5")
    st.markdown("### Classification Report")
    st.code(report)

with tab3:
    st.subheader("🔮 Expense Forecast — Next 3 Months")
    lr = LinearRegression()
    lr.fit(monthly_df[['month']], monthly_df['total'])
    max_m = monthly_df['month'].max()
    future_months = pd.DataFrame({'month': [max_m+1, max_m+2, max_m+3]})
    forecast = lr.predict(future_months)
    labels = ['Month +1', 'Month +2', 'Month +3']

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    fig5, ax5 = style_chart(fig5, ax5)
    ax5.plot(monthly_df['month'], monthly_df['total'],
             marker='o', color=BLUE, label='Actual', linewidth=2.5, markersize=7)
    fx = [monthly_df['month'].iloc[-1]] + list(future_months['month'])
    fy = [monthly_df['total'].iloc[-1]] + list(forecast)
    ax5.plot(fx, fy, marker='s', color=ORANGE, linestyle='--',
             label='Forecast', linewidth=2.5, markersize=7)
    for x, y, l in zip(future_months['month'], forecast, labels):
        ax5.annotate(f'₹{y:,.0f}', (x, y), textcoords='offset points',
                    xytext=(0, 12), ha='center', color=YELLOW, fontweight='bold', fontsize=10)
    ax5.legend(facecolor=SURFACE, edgecolor='#30363d', labelcolor='white')
    ax5.set_ylabel("Amount (₹)")
    plt.tight_layout()
    st.pyplot(fig5)

    fc1, fc2, fc3 = st.columns(3)
    for col, l, v in zip([fc1, fc2, fc3], labels, forecast):
        col.metric(f"🔮 {l}", f"₹{v:,.0f}")

with tab4:
    st.subheader("💬 AI Financial Advisor")
    st.markdown("<p style='color:#8b949e;'>Ask me anything about your finances!</p>",
                unsafe_allow_html=True)

    avg_monthly = monthly_df['total'].mean()
    last_month = monthly_df['total'].iloc[-1]
    budget = avg_monthly * 1.1
    remaining = max(0, budget - last_month)

    ITEM_COSTS = {
        'laptop': 50000, 'phone': 30000, 'clothes': 5000,
        'vacation': 20000, 'trip': 15000, 'iphone': 80000,
        'watch': 10000, 'gym': 2000, 'course': 5000,
        'bike': 80000, 'tablet': 25000, 'headphones': 5000,
        'camera': 40000, 'tv': 35000, 'fridge': 25000
    }

    def advisor(q):
        q_low = q.lower()
        detected_item, detected_cost = None, None
        for item, cost in ITEM_COSTS.items():
            if item in q_low:
                detected_item, detected_cost = item, cost
                break
        nums = re.findall(r'\b(\d{3,6})\b', q_low)
        if nums and not detected_cost:
            detected_cost = int(nums[0])
            detected_item = 'item'

        reply = f"📊 **Your Financial Summary:**\n"
        reply += f"- Average monthly spend: **₹{avg_monthly:,.0f}**\n"
        reply += f"- Last month spent: **₹{last_month:,.0f}**\n"
        reply += f"- Budget remaining: **₹{remaining:,.0f}**\n\n"

        if detected_item and detected_cost:
            reply += f"🛒 **Purchase Analysis: {detected_item.title()} (~₹{detected_cost:,})**\n\n"
            if remaining >= detected_cost:
                reply += f"✅ **YES, you can afford it!**\nAfter purchase, ₹{remaining-detected_cost:,.0f} will remain in your budget."
            elif remaining >= detected_cost * 0.5:
                reply += f"⚠️ **RISKY purchase!**\nYou're short by ₹{detected_cost-remaining:,.0f}. Consider waiting 1 more month to save up."
            else:
                reply += f"❌ **Not recommended right now.**\nYou're short by ₹{detected_cost-remaining:,.0f}. Save for 2-3 months before buying."
        elif any(w in q_low for w in ['save', 'saving', 'cut']):
            saving = avg_monthly * 0.15
            reply += f"💡 **Savings Tip:**\nCut 15% from **{cat_totals.index[0]}** spending.\nThat saves ~₹{saving:,.0f}/month = **₹{saving*12:,.0f}/year!** 🎯"
        elif any(w in q_low for w in ['highest', 'most', 'worst', 'top']):
            reply += f"📌 **Top Spending Categories:**\n"
            for i, (cat, amt) in enumerate(cat_totals.head(3).items()):
                reply += f"{i+1}. **{cat}** — ₹{amt:,.0f}\n"
        elif any(w in q_low for w in ['budget', 'limit']):
            reply += f"💰 **Your Monthly Budget:** ₹{budget:,.0f}\n(10% above your average spend)"
        elif any(w in q_low for w in ['average', 'avg', 'mean']):
            reply += f"📈 **Average Monthly Spending:** ₹{avg_monthly:,.0f}"
        else:
            reply += "💬 **Try asking:**\n"
            reply += "- *Can I afford a laptop?*\n"
            reply += "- *Can I buy a phone for 30000?*\n"
            reply += "- *How can I save money?*\n"
            reply += "- *What is my highest spending category?*\n"
            reply += "- *What is my monthly budget?*"
        return reply

    # Quick buttons
    st.markdown("**💡 Quick Questions:**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    quick_q = None
    with qc1:
        if st.button("💻 Can I buy a laptop?"):
            quick_q = "Can I afford a laptop?"
    with qc2:
        if st.button("💰 How to save money?"):
            quick_q = "How can I save money?"
    with qc3:
        if st.button("📊 Top spending?"):
            quick_q = "What is my highest spending category?"
    with qc4:
        if st.button("📅 My budget?"):
            quick_q = "What is my monthly budget?"

    if quick_q:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": quick_q})
        st.session_state.messages.append({"role": "assistant", "content": advisor(quick_q)})
        st.rerun()

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Type your question here... e.g. Can I afford a phone?"):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        answer = advisor(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
