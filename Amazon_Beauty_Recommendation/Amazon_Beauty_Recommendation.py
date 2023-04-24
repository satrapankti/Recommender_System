import surprise
import pandas as pd
from pickle import dump, load
import streamlit as st
import base64



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-position: 55% 75%;
        background-size: contain;
        background-repeat: no-repeat
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local("amazon.png")




st.title("**Amazon Beauty Recommendation System**")

beautys = pd.read_csv("Amazon_Beauty_Recommendation.csv")

User_Id = st.text_input("Enter your user ID:")
Product_Type = st.selectbox("Select a product category:", beautys["ProductType"].unique())

data =  pd.DataFrame({"User ID":User_Id, "Product Type":Product_Type},index = [0])
st.markdown("**:green[User Input parameters]**")
inp = pd.DataFrame(data)
st.write(inp)




# load the model from disk
model = load(open("Beauty.sav", "rb"))

# Function to recommend products based on user input
# If user is in the list, use recommender system elseif user is not in the list, recommend popular products 
def recommend_products(user_id, product_type):
    if user_id in beautys.UserId:
        product_list = beautys.loc[beautys["ProductType"] == product_type, "ProductId"].unique()
        predictions = [(product_id, model.predict(user_id, product_id).est) for product_id in product_list]
        sorted_predictions = sorted(predictions, key = lambda x: x[1], reverse = True)
        print("Top 5 recommended products for user", user_id, "in the", product_type, "category:")
        for i in range(5):
            product_id = sorted_predictions[i][0]
            url = beautys.loc[beautys["ProductId"] == product_id, "URL"].iloc[0]
            print(i + 1, "- Product ID:", product_id, "\nURL:", url)
    else: 
        top_products = beautys.loc[beautys["ProductType"] == product_type].groupby("ProductId")["Rating"].mean().sort_values(ascending = False).index[:5]
        print("Top 5 popular products in the", product_type, "category:")
        for i, product_id in enumerate(top_products):
            url = beautys.loc[beautys["ProductId"] == product_id, "URL"].iloc[0]
            print(i + 1, "- Product ID:", product_id, "\nURL:", url)




if st.button("Recommend"):
    recommend_products(User_Id, Product_Type)
