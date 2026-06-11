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
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Expense Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Advanced CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

:root {
    --bg:          #0d1117;
    --surface:     #161b22;
    --surface2:    #21262d;
    --border:      #30363d;
    --accent:      #58a6ff;
    --accent-glow: rgba(88,166,255,.15);
    --green:       #3fb950;
    --red:         #f85149;
    --amber:       #d29922;
    --text-1:      #e6edf3;
    --text-2:      #8b949e;
    --radius:      12px;
    --radius-lg:   18px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text-1) !important;
}

.stApp { background: var(--bg) !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-1) !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-1) !important;
}

/* ── Header ── */
header[data-testid="stHeader"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Title ── */
h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #58a6ff 0%, #a371f7 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.5px !important;
}

h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-1) !important;
    font-weight: 600 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.2rem 1.5rem !important;
    transition: transform .15s, box-shadow .15s !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #58a6ff, #a371f7);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,0,0,.5) !important;
}
[data-testid="stMetricLabel"] > div {
    color: var(--text-2) !important;
    font-size: .75rem !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] > div {
    color: var(--text-1) !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 2px !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: .875rem !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
    padding: .6rem 1.2rem !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    border: none !important;
    background: transparent !important;
    transition: color .15s, background .15s !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: var(--text-1) !important;
    background: var(--surface2) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--surface2) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Charts ── */
[data-testid="stImage"] img,
[data-testid="stPyplotRootElement"] img {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border) !important;
}

/* ── Code blocks ── */
[data-testid="stCode"], pre, code {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: #79c0ff !important;
    font-size: .82rem !important;
    line-height: 1.7 !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Success / Info banners ── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface2) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: .65rem !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-1) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    padding: .5rem 1.25rem !important;
    transition: opacity .15s, transform .1s !important;
}
.stButton > button:hover {
    opacity: .85 !important;
    transform: translateY(-1px) !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: var(--text-2) !important;
    font-size: .875rem !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ── Main block padding ── */
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper: apply dark theme to any fig/ax ───────────────────────────────────
def dark_fig(fig, ax, accent_color):
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors=accent_color)
    ax.yaxis.label.set_color(accent_color)
    ax.xaxis.label.set_color(accent_color)
    ax.title.set_color(accent_color)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(True, color='#30363d', alpha=0.4, linewidth=0.6)
    fig.tight_layout()

# Chart accent colors per figure
C1 = '#f9f871'   # fig1 — lemon yellow
C2 = '#89cff0'   # fig2 — baby blue
C3 = '#c3b1e1'   # fig3 — pastel purple
C4 = '#ffb347'   # fig4 — pastel orange
C5 = '#ffb6c1'   # fig5 — pastel pink


# ── App title ────────────────────────────────────────────────────────────────
st.title("📈AI Expense Analyzer Dashboard")

uploaded = st.file_uploader("Upload your Expenses_clean.csv", type="csv")

if uploaded:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
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

    df = load_data(uploaded)
    st.success(f"✅ Loaded {len(df)} transactions successfully!")

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")
        st.divider()
        months = sorted(df['month_name'].unique().tolist())
        selected_months = st.multiselect("Months", months, default=months)
        cats = sorted(df['category'].unique().tolist())
        selected_cats = st.multiselect("Categories", cats, default=cats)
        df = df[df['month_name'].isin(selected_months) & df['category'].isin(selected_cats)]
        st.divider()
        st.info(f"Showing **{len(df)}** transactions")

    # ── Metric cards ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spent",    f"₹ {df['amount'].sum():,.0f}")
    col2.metric("Transactions",   len(df))
    monthly_avg = df.groupby('month')['amount'].sum().mean()
    col3.metric("Avg / Month",    f"₹ {monthly_avg:,.0f}")
    col4.metric("Categories",     df['category'].nunique())

    st.divider()

    # ── Derived data ──────────────────────────────────────────────────────────
    month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                 7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly_df = df.groupby('month')['amount'].sum().reset_index()
    monthly_df.columns = ['month', 'total']
    cat_totals  = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    acct_totals = df.groupby('account')['amount'].sum().sort_values(ascending=False)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "ML Models",
        "Forecast",
        "AI Advisor"
    ])

    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Monthly Spending Trend")
            fig1, ax1 = plt.subplots(figsize=(7, 4))
            dark_fig(fig1, ax1, C1)
            ax1.plot(monthly_df['month'], monthly_df['total'],
                     marker='o', color=C1, linewidth=2.5)
            ax1.fill_between(monthly_df['month'], monthly_df['total'],
                             alpha=0.12, color=C1)
            ax1.set_xticks(monthly_df['month'])
            ax1.set_xticklabels([month_map[m] for m in monthly_df['month']], rotation=45)
            ax1.set_ylabel("Amount (₹)")
            st.pyplot(fig1)

        with c2:
            st.subheader("Spending by Category")
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            fig2.patch.set_facecolor('#161b22')
            top6   = cat_totals.head(6)
            others = cat_totals.iloc[6:].sum()
            vals   = list(top6.values) + ([others] if others > 0 else [])
            lbls   = list(top6.index)  + (['Others'] if others > 0 else [])
            pie_colors = ['#89cff0','#58a6ff','#a371f7','#3fb950','#ffb347','#ffb6c1','#8b949e']
            wedges, texts, autotexts = ax2.pie(
                vals, labels=lbls, autopct='%1.1f%%', startangle=90,
                colors=pie_colors[:len(vals)],
                textprops={'color': C2, 'fontsize': 9}
            )
            for at in autotexts:
                at.set_color('#e6edf3')
                at.set_fontsize(8)
            fig2.tight_layout()
            st.pyplot(fig2)

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Spending by Account")
            fig3, ax3 = plt.subplots(figsize=(7, 3))
            dark_fig(fig3, ax3, C3)
            bar_colors = [C3, '#58a6ff', '#f85149', '#3fb950', '#ffb347']
            ax3.bar(acct_totals.index, acct_totals.values,
                    color=bar_colors[:len(acct_totals)], width=0.55)
            ax3.set_ylabel("Amount (₹)")
            plt.xticks(rotation=20)
            st.pyplot(fig3)

        with c4:
            st.subheader("Spending by Day of Week")
            fig4, ax4 = plt.subplots(figsize=(7, 3))
            dark_fig(fig4, ax4, C4)
            dow       = df.groupby('day_of_week')['amount'].sum()
            day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            ax4.bar(range(len(dow)), dow.values, color=C4, width=0.55)
            ax4.set_xticks(range(len(dow)))
            ax4.set_xticklabels([day_names[i] for i in dow.index])
            ax4.set_ylabel("Amount (₹)")
            st.pyplot(fig4)

    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("ML Model Training & Results")

        @st.cache_resource
        def train(_df):
            le_cat  = LabelEncoder()
            le_acct = LabelEncoder()
            _df = _df.copy()
            _df['category_encoded'] = le_cat.fit_transform(_df['category'])
            _df['account_encoded']  = le_acct.fit_transform(_df['account'])
            X = _df[['amount','month','day_of_week','week','account_encoded']]
            y = _df['category_encoded']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            acc    = accuracy_score(y_test, preds)
            report = classification_report(
                y_test, preds, target_names=le_cat.classes_, zero_division=0)
            return acc, report

        with st.spinner("Training Random Forest model..."):
            acc, report = train(df)

        st.metric("Random Forest Accuracy", f"{acc*100:.1f}%")
        st.text("Classification Report:")
        st.code(report)

    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Expense Forecast")

        lr = LinearRegression()
        lr.fit(monthly_df[['month']], monthly_df['total'])
        future_months = pd.DataFrame({'month': [12, 13, 14]})
        forecast      = lr.predict(future_months)
        labels        = ['Dec 2025', 'Jan 2026', 'Feb 2026']

        fig5, ax5 = plt.subplots(figsize=(10, 4))
        dark_fig(fig5, ax5, C5)
        ax5.plot(monthly_df['month'], monthly_df['total'],
                 marker='o', color=C5, label='Actual', linewidth=2)
        future_x = [monthly_df['month'].iloc[-1], 12, 13, 14]
        future_y = [monthly_df['total'].iloc[-1]] + list(forecast)
        ax5.plot(future_x, future_y,
                 marker='s', color='#f85149', linestyle='--',
                 label='Forecast', linewidth=2)
        for x, y, l in zip([12, 13, 14], forecast, labels):
            ax5.annotate(f'₹{y:.0f}', (x, y),
                         textcoords='offset points', xytext=(0, 12),
                         ha='center', color='#f85149', fontsize=9)
        ax5.legend(facecolor='#21262d', edgecolor='#30363d',
                   labelcolor='#e6edf3', fontsize=9)
        ax5.set_ylabel("Amount (₹)")
        st.pyplot(fig5)

        st.subheader("Forecasted Values")
        fc1, fc2, fc3 = st.columns(3)
        for col, lbl, val in zip([fc1, fc2, fc3], labels, forecast):
            col.metric(lbl, f"₹ {val:,.0f}")

    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("AI Financial Advisor")

        avg_monthly = monthly_df['total'].mean()
        last_month  = monthly_df['total'].iloc[-1]
        budget      = avg_monthly * 1.1
        remaining   = max(0, budget - last_month)

        ITEM_COSTS = {
            'laptop': 800, 'phone': 500, 'clothes': 200,
            'vacation': 1000, 'trip': 800, 'iphone': 900,
            'watch': 300, 'gym': 50, 'course': 150
        }

        def advisor(q):
            import re
            q_low = q.lower()
            detected_item, detected_cost = None, None
            for item, cost in ITEM_COSTS.items():
                if item in q_low:
                    detected_item, detected_cost = item, cost
                    break
            nums = re.findall(r'\b(\d{3,5})\b', q_low)
            if nums and not detected_cost:
                detected_cost = int(nums[0])
                detected_item = 'item'

            reply  = f"Avg monthly spend: **₹ {avg_monthly:,.0f}**\n\n"
            reply += f"Last month: **₹ {last_month:,.0f}** | Remaining budget: **₹ {remaining:,.0f}**\n\n"

            if detected_item and detected_cost:
                reply += f"Purchase detected: **{detected_item.title()}** (~₹ {detected_cost:,})\n\n"
                if remaining >= detected_cost:
                    reply += f"✅ YES you can buy it! ~₹ {remaining-detected_cost:,.0f} will remain."
                elif remaining >= detected_cost * 0.5:
                    reply += f"⚠️RISKY! You're short by ₹ {detected_cost-remaining:,.0f}. Wait 1 more month."
                else:
                    reply += f"❌NOT recommended. Short by ₹ {detected_cost-remaining:,.0f}. Save for 2–3 months."
            elif 'save' in q_low:
                saving = avg_monthly * 0.15
                reply += f"Cut 15% from **{cat_totals.index[0]}** → save ~₹ {saving:,.0f}/month!"
            else:
                reply += "Ask me things like:\n- *Can I afford a laptop?*\n- *What if I buy a phone?*\n- *How can I save money?*"
            return reply

        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if question := st.chat_input("Ask me anything about your finances..."):
            with st.chat_message("user"):
                st.write(question)
            st.session_state.messages.append({"role": "user", "content": question})
            answer = advisor(question)
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ── No file uploaded ──────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 18px;
        padding: 2.5rem 2rem;
        margin-top: 2rem;
        max-width: 640px;
    ">
        <h3 style="color:#58a6ff; font-family:'Space Grotesk',sans-serif; margin-top:0">
            Welcome to AI Expense Analyzer
        </h3>
        <p style="color:#8b949e; line-height:1.7">
            Upload your <code style="background:#21262d;padding:2px 7px;border-radius:6px;color:#79c0ff">Expenses_clean.csv</code>
            file using the uploader above to get started.
        </p>
        <hr style="border-color:#30363d; margin:1.2rem 0"/>
        <p style="color:#8b949e; font-size:.9rem; margin-bottom:.4rem"><strong style="color:#e6edf3">What you'll get:</strong></p>
        <p style="color:#8b949e; font-size:.9rem; line-height:2">
            <strong style="color:#e6edf3">Overview</strong> — Monthly trends, category & account breakdown<br>
            <strong style="color:#e6edf3">ML Models</strong> — Random Forest predicts your spending category<br>
            <strong style="color:#e6edf3">Forecast</strong> — Predicts your next 3 months of expenses<br>
            <strong style="color:#e6edf3">AI Advisor</strong> — Chat to get personalized financial advice
        </p>
    </div>
    """, unsafe_allow_html=True)
