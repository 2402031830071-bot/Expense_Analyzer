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

st.set_page_config(page_title="AI Expense Analyzer", page_icon="💰", layout="wide")
st.title(" AI Expense Analyzer Dashboard")

uploaded = st.file_uploader("Upload your Expenses_clean.csv", type="csv")

if uploaded:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        df.rename(columns={'date_time':'date'}, inplace=True)
        df.drop(columns=['currency','tags'], inplace=True, errors='ignore')
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

    df = load_data(uploaded)
    st.success(f"Loaded {len(df)} transactions!")

    # ── Sidebar filters ──
    with st.sidebar:
        st.header(" Filters")
        months = sorted(df['month_name'].unique().tolist())
        selected_months = st.multiselect("Months", months, default=months)
        cats = sorted(df['category'].unique().tolist())
        selected_cats = st.multiselect("Categories", cats, default=cats)
        df = df[df['month_name'].isin(selected_months) & df['category'].isin(selected_cats)]
        st.info(f"Showing {len(df)} transactions")

    # ── Metric cards ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spent", f"{df['amount'].sum():.0f} BYN")
    col2.metric(" Transactions", len(df))
    monthly_avg = df.groupby('month')['amount'].sum().mean()
    col3.metric(" Avg/Month", f"{monthly_avg:.0f} BYN")
    col4.metric(" Categories", df['category'].nunique())

    st.divider()

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([" Overview","ML Models"," Forecast"," AI Advisor"])

    month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                 7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly_df = df.groupby('month')['amount'].sum().reset_index()
    monthly_df.columns = ['month','total']
    cat_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    acct_totals = df.groupby('account')['amount'].sum().sort_values(ascending=False)

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Monthly Spending Trend")
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(monthly_df['month'], monthly_df['total'], marker='o', color='#3498db', linewidth=2.5)
            ax.fill_between(monthly_df['month'], monthly_df['total'], alpha=0.15, color='#3498db')
            ax.set_xticks(monthly_df['month'])
            ax.set_xticklabels([month_map[m] for m in monthly_df['month']], rotation=45)
            ax.set_ylabel("Amount (BYN)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        with c2:
            st.subheader("Spending by Category")
            fig2, ax2 = plt.subplots(figsize=(7,4))
            top6 = cat_totals.head(6)
            others = cat_totals.iloc[6:].sum()
            vals = list(top6.values) + [others]
            lbls = list(top6.index) + ['Others']
            ax2.pie(vals, labels=lbls, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig2)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Spending by Account")
            fig3, ax3 = plt.subplots(figsize=(7,3))
            ax3.bar(acct_totals.index, acct_totals.values, color=['#9b59b6','#3498db','#e74c3c'])
            ax3.set_ylabel("Amount (BYN)")
            st.pyplot(fig3)
        with c4:
            st.subheader("Spending by Day of Week")
            fig4, ax4 = plt.subplots(figsize=(7,3))
            dow = df.groupby('day_of_week')['amount'].sum()
            day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            ax4.bar(range(len(dow)), dow.values, color='#e67e22')
            ax4.set_xticks(range(len(dow)))
            ax4.set_xticklabels([day_names[i] for i in dow.index])
            ax4.set_ylabel("Amount (BYN)")
            st.pyplot(fig4)

    with tab2:
        st.subheader(" ML Model Training & Results")
        @st.cache_resource
        def train(_df):
            le_cat  = LabelEncoder()
            le_acct = LabelEncoder()
            _df = _df.copy()
            _df['category_encoded'] = le_cat.fit_transform(_df['category'])
            _df['account_encoded']  = le_acct.fit_transform(_df['account'])
            X = _df[['amount','month','day_of_week','week','account_encoded']]
            y = _df['category_encoded']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds,
                         target_names=le_cat.classes_, zero_division=0)
            return acc, report

        with st.spinner("Training ML model..."):
            acc, report = train(df)

        st.metric("Random Forest Accuracy", f"{acc*100:.1f}%")
        st.text("Classification Report:")
        st.code(report)

    with tab3:
        st.subheader(" Expense Forecast")
        lr = LinearRegression()
        lr.fit(monthly_df[['month']], monthly_df['total'])
        future_months = pd.DataFrame({'month':[12,13,14]})
        forecast = lr.predict(future_months)
        labels = ['Dec 2025','Jan 2026','Feb 2026']

        fig5, ax5 = plt.subplots(figsize=(10,4))
        ax5.plot(monthly_df['month'], monthly_df['total'],
                 marker='o', color='#3498db', label='Actual', linewidth=2)
        future_x = [monthly_df['month'].iloc[-1], 12, 13, 14]
        future_y = [monthly_df['total'].iloc[-1]] + list(forecast)
        ax5.plot(future_x, future_y,
                 marker='s', color='#e74c3c', linestyle='--', label='Forecast', linewidth=2)
        for x, y, l in zip([12,13,14], forecast, labels):
            ax5.annotate(f'{y:.0f}', (x,y), textcoords='offset points', xytext=(0,10),
                        ha='center', color='red')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylabel("Amount (BYN)")
        st.pyplot(fig5)

        st.subheader("Forecasted Values")
        for l, v in zip(labels, forecast):
            st.write(f"**{l}** → {v:.0f} BYN")

    with tab4:
        st.subheader("💬 AI Financial Advisor")
        avg_monthly = monthly_df['total'].mean()
        last_month  = monthly_df['total'].iloc[-1]
        budget      = avg_monthly * 1.1
        remaining   = max(0, budget - last_month)

        ITEM_COSTS = {
            'laptop':800,'phone':500,'clothes':200,
            'vacation':1000,'trip':800,'iphone':900,
            'watch':300,'gym':50,'course':150
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

            reply = f"Your avg monthly spend: **{avg_monthly:.0f} BYN**\n\n"
            reply += f" Last month: **{last_month:.0f} BYN** | Remaining budget: **{remaining:.0f} BYN**\n\n"

            if detected_item and detected_cost:
                reply += f" Purchase detected: **{detected_item.title()}** (~{detected_cost} BYN)\n\n"
                if remaining >= detected_cost:
                    reply += f" YES you can buy it! ~{remaining-detected_cost:.0f} BYN will remain."
                elif remaining >= detected_cost * 0.5:
                    reply += f" RISKY! You're short by {detected_cost-remaining:.0f} BYN. Wait 1 more month."
                else:
                    reply += f" NOT recommended. Short by {detected_cost-remaining:.0f} BYN. Save for 2-3 months."
            elif 'save' in q_low:
                saving = avg_monthly * 0.15
                reply += f" Cut 15% from **{cat_totals.index[0]}** → save ~{saving:.0f} BYN/month!"
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
            st.session_state.messages.append({"role":"user","content":question})
            answer = advisor(question)
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role":"assistant","content":answer})

else:
    st.info("Please upload your Expenses_clean.csv file to start!")
    st.markdown("""
    ### What this dashboard does:
    - **Overview** — Monthly trends, category breakdown, account analysis
    -  **ML Models** — Random Forest predicts your spending category
    -  **Forecast** — Predicts your next 3 months of expenses
    - **AI Advisor** — Chat to get personalized financial advice
    """)
