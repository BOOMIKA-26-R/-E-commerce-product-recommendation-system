import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Product Recommender", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Ecommerce_Personalized_Recommendation_Dataset.csv') 
        df = df.dropna(subset=['User_ID', 'Product_ID', 'User_Rating'])
        return df
    except Exception as e:
        st.error(f"Error: Make sure your file is renamed to 'Ecommerce_Personalized_Recommendation_Dataset.csv' on GitHub. ({e})")
        return None

@st.cache_data
def get_similarity_matrix(df):
    # Create Matrix
    pivot = df.pivot_table(index='User_ID', columns='Product_ID', values='User_Rating', fill_value=0)
    # Calculate Cosine Similarity
    item_sim = cosine_similarity(pivot.T)
    return pd.DataFrame(item_sim, index=pivot.columns, columns=pivot.columns)

st.title("🛍️ AI Product Recommendation Dashboard")
st.markdown("---")

df = load_data()

if df is not None:
    matrix = get_similarity_matrix(df)

    with st.sidebar:
        st.header("📊 Global Analytics")
        st.metric("Total Items", df['Product_ID'].nunique())
        st.metric("Total Users", df['User_ID'].nunique())
        st.metric("Avg Rating", round(df['User_Rating'].mean(), 2))

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Select Product")
        selected_prod = st.selectbox("Search Product ID:", df['Product_ID'].unique())
        btn = st.button("Generate Recommendations")

    with col2:
        if btn:
            scores = matrix[selected_prod].sort_values(ascending=False).iloc[1:6]
            
            st.subheader("🚀 Top 5 AI Suggestions")
            res_cols = st.columns(5)
            for i, (p_id, score) in enumerate(scores.items()):
                with res_cols[i]:
                    st.success(f"**{p_id}**")
                    st.caption(f"Match: {round(score*100, 1)}%")

            st.write("---")
            st.subheader("📈 Recommendation Similarity Strength")
            
            st.bar_chart(scores)
            
            st.info("The chart above shows the 'Cosine Similarity' score. A higher bar means the product is more closely related to your selection.")

st.sidebar.markdown("---")
st.sidebar.caption("System Status: Online 🟢")
