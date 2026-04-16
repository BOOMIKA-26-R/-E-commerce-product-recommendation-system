import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="E-comm AI Dashboard", layout="wide")

@st.cache_data
def load_data():
    current_dir = Path(__file__).parent
    file_path = current_dir / "Ecommerce_Personalized_Recommendation_Dataset.csv"
    
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['User_ID', 'Product_ID', 'User_Rating'])
        return df
    except FileNotFoundError:
        st.error(f"File not found at: {file_path}")
        return None

@st.cache_data
def get_recommendation_engine(df):

    pivot = df.pivot_table(index='User_ID', columns='Product_ID', values='User_Rating', fill_value=0)

    item_sim = cosine_similarity(pivot.T)
    item_sim_df = pd.DataFrame(item_sim, index=pivot.columns, columns=pivot.columns)
    return item_sim_df

st.title("🛍️ E-commerce Product Recommendation System")
st.markdown("---")

try:
    data = load_data()
    engine = get_recommendation_engine(data)

    st.sidebar.header("📊 Global Analytics")
    st.sidebar.metric("Total Products", data['Product_ID'].nunique())
    st.sidebar.metric("Total Users", data['User_ID'].nunique())
    st.sidebar.metric("Avg Platform Rating", round(data['User_Rating'].mean(), 2))

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Select a Product")
        selected_prod = st.selectbox("Search Product ID:", data['Product_ID'].unique())
        
        # Show details of selected product
        prod_info = data[data['Product_ID'] == selected_prod].iloc[0]
        st.write(f"**Current Product:** {selected_prod}")
        st.write(f"**Category:** {prod_info.get('Product_Category', 'General')}")
        
        btn = st.button("Generate AI Recommendations")

    with col2:
        if btn:
            similar_scores = engine[selected_prod].sort_values(ascending=False).iloc[1:6]
            recs = similar_scores.index.tolist()

            st.subheader("🚀 Top 5 AI Recommendations")
            
            rec_cols = st.columns(5)
            for i, p_id in enumerate(recs):
                with rec_cols[i]:
                    st.info(f"**{p_id}**")
                    score = round(similar_scores[i] * 100, 1)
                    st.caption(f"Match: {score}%")

            st.write("---")
            st.subheader("📈 Recommendation Strength")
            st.bar_chart(similar_scores)

except Exception as e:
    st.error(f"Please ensure 'Ecommerce_Personalized_Recommendation_Dataset.csv' is in the folder. Error: {e}")
