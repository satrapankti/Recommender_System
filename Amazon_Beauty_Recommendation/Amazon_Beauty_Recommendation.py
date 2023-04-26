import io
import bz2
import base64
import surprise
import urllib.request
import pandas as pd
import streamlit as st


st.title("**Amazon Beauty Recommendation System**")


def add_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://github.com/satrapankti/Recommender_System/blob/main/Amazon_Beauty_Recommendation/Beauty.png?raw=true");
            background-position: 55% 75%;
            background-size: contain;
            background-repeat: no-repeat
            }}
            </style>
            """,
        unsafe_allow_html=True
    )
#add_bg()


@st.cache
def load_data():
    url = "https://github.com/satrapankti/Recommender_System/raw/main/Amazon_Beauty_Recommendation/Amazon_Beauty_Recommendation.bz2"
    response = urllib.request.urlopen(url)
    compressed_data = response.read()

    decompressed_data = bz2.decompress(compressed_data)
    data = decompressed_data.decode()

    data = [d.split(",") for d in data.split("\n")]
    data = pd.DataFrame(data[1:], columns=data[0])
    return data

beauty = load_data()



# Loading and splitting the data
reader = Reader(rating_scale = (1, 5))
beauty_data = Dataset.load_from_df(beauty[["UserId", "ProductId", "Rating"]], reader)


User_Id = st.text_input("Enter your user ID:")
Product_Type = st.selectbox("Select a product category:", beauty["ProductType"].unique())

data =  pd.DataFrame({"User ID":User_Id, "Product Type":Product_Type},index = [0])
st.markdown("**:green[User Input parameters]**")
inp = pd.DataFrame(data)
st.write(inp)


# Function to recommend products based on user input
# If user is in the list, use recommender system elseif user is not in the list, recommend popular products 
def recommend_products(user_id, product_type):
    if user_id in beauty.UserId:
        product_list = beauty.loc[beauty["ProductType"] == product_type, "ProductId"].unique()
        predictions = [(product_id, model.predict(user_id, product_id).est) for product_id in product_list]
        sorted_predictions = sorted(predictions, key = lambda x: x[1], reverse = True)
        print("Top 5 recommended products for user", user_id, "in the", product_type, "category:")
        for i in range(5):
            product_id = sorted_predictions[i][0]
            url = beauty.loc[beauty["ProductId"] == product_id, "URL"].iloc[0]
            print(i + 1, "- Product ID:", product_id, "\nURL:", url)
    else: 
        top_products = beauty.loc[beauty["ProductType"] == product_type].groupby("ProductId")["Rating"].mean().sort_values(ascending = False).index[:5]
        print("Top 5 popular products in the", product_type, "category:")
        for i, product_id in enumerate(top_products):
            url = beauty.loc[beauty["ProductId"] == product_id, "URL"].iloc[0]
            print(i + 1, "- Product ID:", product_id, "\nURL:", url)


model = SVD(n_factors = 50, reg_all = 0.02, lr_all = 0.005, n_epochs = 20)
model.fit(beauty_data.build_full_trainset())


if st.button("Recommend"):
    recommend_products(User_Id, Product_Type)
