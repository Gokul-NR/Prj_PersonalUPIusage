import streamlit as st
import requests
import pdfplumber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import time

# Gemini API settings
# NOTE: The API key is a placeholder. You will need to replace this
# with your actual Gemini API key from the Google AI Studio or Google Cloud.
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="
API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

st.set_page_config(page_title="Finance Assistant", layout="wide")
st.title("💰 Finance Assistant")

# --- Upload PDF ---
st.info("Please upload a PDF file to view the insights and get advise.")
uploaded_file = st.file_uploader("Upload Bank Statement (PDF)", type="pdf")
bank_text = ""
financial_summary = ""

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            bank_text += page.extract_text() + "\n"

    try:
        data = []
        lines = bank_text.split("\n")
        
        # New parsing logic to handle multi-line descriptions
        transactions = []
        current_transaction = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if the line starts with a date pattern (e.g., dd/mm/yyyy)
            date_match = re.match(r'^\d{2}/\d{2}/\d{4}', line)
            
            if date_match:
                # If there's a previous transaction, add it to the list
                if current_transaction:
                    transactions.append(current_transaction)
                
                # Start a new transaction
                parts = line.split()
                try:
                    # Find the index of the first numerical value (amount or balance)
                    first_num_idx = -1
                    for i, part in enumerate(parts):
                        try:
                            float(part.replace(',', ''))
                            first_num_idx = i
                            break
                        except ValueError:
                            continue
                    
                    if first_num_idx != -1 and first_num_idx + 1 < len(parts):
                        date = parts[0]
                        amount = float(parts[first_num_idx].replace(',', ''))
                        balance = float(parts[first_num_idx + 1].replace(',', ''))
                        description = " ".join(parts[1:first_num_idx])
                        
                        current_transaction = {
                            "date": date,
                            "description": description,
                            "amount": amount,
                            "balance": balance
                        }
                    else:
                        # Handle lines that start with a date but don't have the expected numeric format
                        current_transaction = None
                except (ValueError, IndexError):
                    current_transaction = None
            else:
                # This is a continuation of the description from the previous line
                if current_transaction:
                    current_transaction["description"] += " " + line

        # Add the last transaction to the list
        if current_transaction:
            transactions.append(current_transaction)
            
        if transactions:
            data = []
            for t in transactions:
                lower_desc = t["description"].lower()
                if any(keyword in lower_desc for keyword in ["incoming paynow", "interest earned"]):
                    amount = abs(t["amount"])
                else:
                    amount = -abs(t["amount"])
                
                data.append([t["date"], t["description"], amount])
            
            df = pd.DataFrame(data, columns=["Date", "Description", "Amount"])
            
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
            df.dropna(subset=["Date"], inplace=True)
            df["Description"] = df["Description"].astype(str)
            df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
            
            # Categorization logic
            categories = {
                "Groceries": ["kalam mart", "sri vaari", "shengsiong", "ntuc fp", "mohd mustafa"],
                "Food": ["briyani", "ijooz", "food panda", "restaurant", "anjappar"],
                "Transport": ["bus/mrt"],
                "Entertainment": ["cineplexes"],
                "Rent": ["rent"],
                "Personal care": ["beautie"]
            }

            def categorize(desc):
                d = desc.lower()
                for cat, keywords in categories.items():
                    if any(k in d for k in keywords):
                        return cat
                return "Others"

            df["Category"] = df["Description"].apply(categorize)

            # Commented out the display of the data frame
            # st.subheader("Generated DataFrame from PDF")
            # st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error processing the PDF file: {e}")
        df = pd.DataFrame() # Create an empty df to prevent errors later

st.subheader("User Goals")
goal_amount = st.number_input("Goal Amount ($)", min_value=1000.0)
timeline = st.number_input("Timeline (months)", min_value=1)

if st.button("Get Advice"):
    if 'df' not in locals() or df.empty:
        st.warning("Please upload a transaction file first.")
    else:
        with st.spinner('Processing data and generating charts...'):
            time.sleep(2)  # Delay for 2 seconds as requested

            df["Month"] = df["Date"].dt.to_period("M")
            
            expenses_df = df[df["Amount"] < 0].copy()
            income_df = df[df["Amount"] > 0].copy()
            
            summary_df = df.groupby("Category")["Amount"].sum().reset_index()
            monthly_summary = df.groupby("Month")["Amount"].sum().reset_index()
            
            wasteful = expenses_df[expenses_df["Category"].isin(["Entertainment", "Food", "Personal care"])].copy()
            transport_wasteful = expenses_df[
                (expenses_df["Category"] == "Transport") &
                (~expenses_df["Description"].str.lower().str.contains("bus|mrt"))
            ].copy()
            wasteful_df = pd.concat([wasteful, transport_wasteful])
            wasteful_summary = wasteful_df.groupby("Category")["Amount"].sum().reset_index()
            
            income_total = income_df["Amount"].sum()
            expense_total = abs(expenses_df["Amount"].sum())
            savings_total = income_total - expense_total
            monthly_income = income_df.groupby("Month")["Amount"].sum()
            monthly_expense = abs(expenses_df.groupby("Month")["Amount"].sum())
            monthly_savings = monthly_income.reindex(monthly_expense.index, fill_value=0) - monthly_expense.reindex(monthly_income.index, fill_value=0)
            
            st.subheader("📊 Financial Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Category-wise Summary")
                if not summary_df.empty:
                    fig, ax = plt.subplots(figsize=(6,6))
                    colors = plt.cm.viridis(np.linspace(0, 1, len(summary_df)))
                    ax.bar(summary_df["Category"], abs(summary_df["Amount"]), color=colors)
                    ax.set_xlabel("Category")
                    ax.set_ylabel("Amount ($)")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.info("No data found for category summary.")
            
            with col2:
                st.write("Wasteful Transactions")
                if not wasteful_summary.empty:
                    fig, ax = plt.subplots(figsize=(6,6))
                    colors = plt.cm.plasma(np.linspace(0, 1, len(wasteful_summary)))
                    ax.bar(wasteful_summary["Category"], abs(wasteful_summary["Amount"]), color=colors)
                    ax.set_xlabel("Category")
                    ax.set_ylabel("Amount ($)")
                    st.pyplot(fig)
                else:
                    st.info("No wasteful transactions found.")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.write("Income vs Expenses")
                if income_total > 0 or expense_total > 0:
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.bar(["Income", "Expenses"], [income_total, expense_total], color=["green", "red"])
                    ax.set_ylabel("Amount ($)")
                    st.pyplot(fig)
                else:
                    st.info("No income or expense data found.")
            
            with col4:
                st.write("Monthly Savings")
                if not monthly_savings.empty and monthly_savings.sum() != 0:
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.plot(monthly_savings.index.astype(str), monthly_savings.values, marker="o", color="blue")
                    ax.set_xlabel("Month")
                    ax.set_ylabel("Savings ($)")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.info("No monthly savings data to plot.")
            
            financial_summary = f"Income: ${income_total:.2f}, Expenses: ${expense_total:.2f}, Savings: ${savings_total:.2f}\n"
            for _, row in summary_df.iterrows():
                financial_summary += f"- {row['Category']}: ${abs(row['Amount']):.2f}\n"

        user_prompt = f"""
        You are a financial advisor AI.
        Your job is to analyze the user's financial summary and provide actionable advice.
        
        User's financial summary:
        {financial_summary}
        
        User's goal:
        Save ${goal_amount} within {timeline} months.
        
        Please provide advice in the following areas:
        
        1. **Monthly Budget Plan**
        - Create a simple monthly budget plan that balances income, expenses, and savings.
        - Show how much should ideally be allocated to needs, wants, and savings each month.
        
        2. **Suggestions to Reduce Expenses**
        - Based on the financial summary categories, identify unnecessary or high-spending areas.
        - Suggest practical steps to reduce spending in these categories so that the user can achieve the financial goal within the given timeline.
        
        3. **Low-risk Investment Options**
        - Recommend safe, low-risk investment opportunities (e.g., fixed deposits, high-yield savings accounts, government bonds, ETFs).
        - Suggest how these investments can help accelerate reaching the goal within the timeline.
        
        ⚠️ Important:
        - Use only the data provided in {{financial_summary}}, {{goal_amount}}, and {{timeline}}.
        - Do not invent numbers outside the given data.
        - Keep your answer clear, structured, and actionable.
        - Present your full response in **bullet points** under each of the three sections.
        """
        
        payload = {
            "contents": [{
                "parts": [{ "text": user_prompt }]
            }],
            "systemInstruction": {
                "parts": [{
                    "text": "You are a world-class financial advisor. Provide a clear, actionable, and friendly financial advice. Focus on practical steps and insights derived from the user's data. Format your response with clear headings and bullet points."
                }]
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                with st.spinner(f'Getting financial advice from Gemini AI... (Attempt {retries + 1}/{max_retries})'):
                    response = requests.post(f"{API_URL}{API_KEY}", json=payload, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    generated_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    st.subheader("💡 Financial Advice")
                    st.write(generated_text)
                    break  # Break out of the loop on success
                elif response.status_code == 503:
                    st.warning(f"Server busy (503). Retrying in 5 seconds... (Attempt {retries + 1}/{max_retries})")
                    time.sleep(5)
                    retries += 1
                else:
                    st.error(f"Error: Could not get advice from Gemini AI. Status Code: {response.status_code}. Response: {response.text}")
                    break  # Break on other errors
            
            except requests.exceptions.RequestException as e:
                st.error(f"A network error occurred: {e}")
                st.info("Please check your internet connection and API credentials.")
                break  # Break on network errors
        
        if retries == max_retries:
            st.error("Maximum retries reached. The model is still unavailable. Please try again later.")
