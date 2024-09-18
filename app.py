from flask import Flask, request, jsonify, render_template
import pandas as pd
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize Flask app
app = Flask(__name__)

# Load the CSV files into dataframes
customers = pd.read_csv('data/customers_indian2.csv')
orders = pd.read_csv('data/orders_indian2.csv')
products = pd.read_csv('data/products_indian.csv')
stores = pd.read_csv('data/stores_indian2.csv')

# Gemini API Key (Replace with your actual Gemini API key)
gemini_api_key = 'AIzaSyDsBK4F8SDQ-VJ36OGPPhB_eEEZt2DmLCE'

# Initialize Gemini (assuming OpenAI class is used for Gemini)
llm = OpenAI(temperature=0.5, openai_api_key=gemini_api_key)

# Helper function to display the menu options
def display_menu():
    return (
        "Welcome to RetailX Assistant! How can I assist you today?\n"
        "1. Check Product Availability\n"
        "2. Track Order\n"
        "3. Find Nearest Store\n"
        "4. Get Personalized Recommendations\n"
        "5. Customer Support\n"
        "Please type the number of the service you need."
    )

# Function to handle the LLM response
def generate_response(prompt):
    prompt_template = PromptTemplate.from_template(prompt)
    chain = LLMChain(prompt_template=prompt_template, llm=llm)
    response = chain.run({})
    return response

# Function to clean price string and convert it to float
def clean_price(price_str):
    price_str = price_str.replace('?', '').replace(',', '').strip()
    return float(price_str)

# Function to check product availability
def check_product_availability(product_name):
    product_info = products[products['ProductName'].str.contains(product_name, case=False)]
    if not product_info.empty:
        product_details = product_info.iloc[0]
        stock_message = f"{product_details['ProductName']} is available with {product_details['Stock']} units in stock."
        return stock_message
    return "Sorry, this product is not available."

# Function to track order status using ProductID
def track_order(product_id):
    try:
        product_id = int(product_id)
    except ValueError:
        return "Invalid ProductID. Please enter a numeric ProductID."
    
    order_info = orders[orders['ProductID'] == product_id]
    
    if not order_info.empty:
        order_details = order_info.iloc[0]
        return (
            f"Order Details for ProductID {product_id}:\n"
            f"Quantity: {order_details['Quantity']}\n"
            f"Order Date: {order_details['OrderDate']}\n"
            f"Status: {order_details['Status']}\n"
        )
    return "No order found with the provided ProductID."

# Function to find the nearest store based on city
def find_nearest_store(city_name):
    store_info = stores[stores['City'].str.contains(city_name, case=False)]
    if not store_info.empty:
        store_details = store_info.iloc[0]
        return (
            f"Nearest store:\n"
            f"Store Name: {store_details['StoreName']}\n"
            f"Address: {store_details['Address']}, {store_details['City']}, {store_details['State']}\n"
            f"Phone: {store_details['Phone']}\n"
            f"Working Hours: {store_details['Hours']}"
        )
    return f"No store found in {city_name}."

# Function to get customer-specific categories for recommendations
def get_customer_categories():
    categories = products['Category'].unique()
    categories_list = ", ".join(categories)
    return f"Here are the available categories:\n{categories_list}"

# Function to get products closest to the price in a specified category
def get_closest_products(category, target_price):
    try:
        target_price = clean_price(target_price)
    except ValueError:
        return "Invalid price. Please provide a numeric value for the price."
    
    filtered_products = products[products['Category'].str.contains(category, case=False)]
    
    if filtered_products.empty:
        return "No products found in this category."
    
    filtered_products['Price'] = filtered_products['Price'].apply(clean_price)
    filtered_products['PriceDiff'] = abs(filtered_products['Price'] - target_price)
    closest_products = filtered_products.sort_values('PriceDiff').head(5)
    
    if closest_products.empty:
        return "No products found close to your specified price."
    
    recommendations_list = []
    for _, product in closest_products.iterrows():
        recommendations_list.append(
            f"ProductID: {product['ProductID']}\n"
            f"ProductName: {product['ProductName']}\n"
            f"Description: {product['Description']}\n"
            f"Price: ₹{product['Price']}\n"
            f"Stock: {product['Stock']}\n"
            f"Category: {product['Category']}"
        )
    
    return "Here are the products closest to your specified price:\n" + "\n\n".join(recommendations_list)

# Function to log customer support inquiries to a CSV
def customer_support_log(customer_id, inquiry):
    support_log = pd.DataFrame([[customer_id, inquiry]], columns=['CustomerID', 'Inquiry'])
    support_log.to_csv('customer_support_log.csv', mode='a', header=False, index=False)
    return "Your inquiry has been logged. Our team will get back to you soon."

# Route to render the chatbot UI with the GET method
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Modify the chatbot function for updated conversation flow
@app.route('/chat', methods=['POST'])
def chatbot():
    data = request.json
    user_input = data.get('user_input')
    session = data.get('session', {})  # to store conversation state
    print(user_input)
    print(session)

    if user_input == "1":
        response = "Please provide the product name."
        session['context'] = 'product_availability'
    elif session.get('context') == 'product_availability':
        response = check_product_availability(user_input)
        response += "\nDo you want to check another product? (Yes/No)"
        session['context'] = 'check_another_product'
    elif session.get('context') == 'check_another_product':
        if user_input.lower() == 'yes':
            response = "Please provide the product name."
            session['context'] = 'product_availability'
        else:
            response = display_menu()
            session['context'] = None

    elif user_input == "2":
        response = "Please provide your ProductID to track your order."
        session['context'] = 'track_order'
    elif session.get('context') == 'track_order':
        product_id = user_input
        response = track_order(product_id)
        response += "\nDo you want to check another order? (Yes/No)"
        session['context'] = 'check_another_order'
    elif session.get('context') == 'check_another_order':
        if user_input.lower() == 'yes':
            response = "Please provide your ProductID to track your order."
            session['context'] = 'track_order'
        else:
            response = display_menu()
            session['context'] = None

    elif user_input == "3":
        response = "Please provide your City to find the nearest store."
        session['context'] = 'find_nearest_store'
    elif session.get('context') == 'find_nearest_store':
        city_name = user_input
        response = find_nearest_store(city_name)
        response += "\nDo you want to check another location? (Yes/No)"
        session['context'] = 'check_another_location'
    elif session.get('context') == 'check_another_location':
        if user_input.lower() == 'yes':
            response = "Please provide your City to find the nearest store."
            session['context'] = 'find_nearest_store'
        else:
            response = display_menu()
            session['context'] = None

    elif user_input == "4":
        response = get_customer_categories()
        response += "\nPlease type the category you're interested in."
        session['context'] = 'select_category'
    elif session.get('context') == 'select_category':
        session['category'] = user_input
        response = "Please provide your approximate price for recommendations."
        session['context'] = 'select_price'
    elif session.get('context') == 'select_price':
        response = get_closest_products(session['category'], user_input)
        response += "\nDo you want to check another category or price? (Yes/No)"
        session['context'] = 'check_another_category_or_price'
    elif session.get('context') == 'check_another_category_or_price':
        if user_input.lower() == 'yes':
            response = get_customer_categories()
            response += "\nPlease type the category you're interested in."
            session['context'] = 'select_category'
        else:
            response = display_menu()
            session['context'] = None

    elif user_input == "5":
        response = "Please provide your CustomerID and the nature of your support inquiry."
        session['context'] = 'customer_support'
    elif session.get('context') == 'customer_support':
        customer_id, inquiry = user_input.split(',', 1)
        response = customer_support_log(customer_id, inquiry)
        response += "\nDo you have another inquiry? (Yes/No)"
        session['context'] = 'check_another_inquiry'
    elif session.get('context') == 'check_another_inquiry':
        if user_input.lower() == 'yes':
            response = "Please provide your CustomerID and the nature of your support inquiry."
            session['context'] = 'customer_support'
        else:
            response = display_menu()
            session['context'] = None


    else:
        response = display_menu()
        session['context'] = None

    return jsonify({'response': response, 'session': session})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)