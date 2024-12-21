import sqlite3
from langchain.prompts import SystemMessagePromptTemplate
from sentence_transformers import SentenceTransformer
import faiss
import os
import re
import subprocess
import warnings
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_core._api.deprecation import LangChainDeprecationWarning

# Ignore LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Set up your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'openai-api-key-here'

def extract_python_code(response):
    """
    Extract Python code from a response text.

    :param response: Text response containing Python code.
    :return: Extracted Python code as a string.
    """
    # Extract code block using regular expressions
    code_match = re.search(r'```python(.*?)```', response, re.DOTALL)
    code_match = re.search(r'```python(.*?)```', response, re.DOTALL)

    if code_match:
        return code_match.group(1).strip()
    else:
        raise ValueError("No Python code found in the response.")


def execute_code_and_generate_plot(code):
    """
    Execute the provided Python code and generate the plot image.

    :param code: String containing Python code to execute.
    :return: Path to the generated image file.
    """
    # Write the code to a temporary file
    code_file_path = 'temp_plot_code.py'
    with open(code_file_path, 'w') as file:
        file.write(code)

    # Execute the code
    try:
        subprocess.run(['python', code_file_path], check=True)
        image_path = 'plot.png'
    except subprocess.CalledProcessError as e:
        print(f"Error executing plot code: {e}")
        image_path = None

    # Clean up temporary files
    os.remove(code_file_path)

    return image_path

# Load and format SQL data
def load_sql_data(db_path, queries):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    data = []
    for query in queries:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        data.extend([dict(zip(columns, row)) for row in rows])

    connection.close()
    return data

def format_data_for_chunks(data):
    """
    Format product and review data into text.
    """
    formatted_texts = []

    product_text = ""
    review_text = ""

    for item in data:
        if 'product_name' in item:
            product_text += (f"Product ID: {item.get('product_id', '')}\n"
                             f"Product Name: {item.get('product_name', '')}\n"
                             f"Description: {item.get('description', '')}\n"
                             f"Price: {item.get('price', '')}\n"
                             f"Category: {item.get('category', '')}\n"
                             f"Brand: {item.get('brand', '')}\n"
                             f"Stock Quantity: {item.get('stock_quantity', '')}\n"
                             f"Rating: {item.get('rating', '')}\n"
                             f"Release Date: {item.get('release_date', '')}\n"
                             f"Supplier: {item.get('supplier', '')}\n\n")
        elif 'comment' in item:
            review_text += (f"Review ID: {item.get('review_id', '')}\n"
                            f"Product ID: {item.get('product_id', '')}\n"
                            f"Rating: {item.get('rating', '')}\n"
                            f"Comment: {item.get('comment', '')}\n\n")

    if product_text:
        formatted_texts.append(product_text)
    if review_text:
        formatted_texts.append(review_text)

    return formatted_texts

def store_data_in_vector_db(model, data):
    """
    Encode and store the entire data in a vector database.
    """
    vectors = model.encode(data)
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    # index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW with 32 neighbors
    index.add(vectors)

    return index

def query_vector_db(prompt, model, index, data, k=3):
    """

    Query the vector database and retrieve relevant data.
    """
    prompt_vector = model.encode([prompt])
    faiss.normalize_L2(prompt_vector)

    distances, indices = index.search(prompt_vector, k)
    top_results = [data[idx] for idx in indices[0]]

    # top_result = data[indices[0][0]] if indices[0].size > 0 else None

    return top_results

if __name__ == "__main__":
    db_path = 'ecommerce.db'
    plot_keywords = ['plot', 'generate', 'chart', 'graph', 'visualize']
    with open('prompt.txt', 'r') as f:
        system = f.read()
    queries = [
        'SELECT product_id, product_name, description, price, category, brand, stock_quantity, rating, release_date, supplier FROM products',
        'SELECT review_id, product_id, rating, comment FROM reviews'
    ]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index_path = "sql_data_index.faiss"

    # Load SQL data
    data = load_sql_data(db_path, queries)
    formatted_data = format_data_for_chunks(data)

    # Store in vector database
    index = store_data_in_vector_db(model, formatted_data)

    # Save the index
    faiss.write_index(index, faiss_index_path)
    print(f"Stored {len(formatted_data)} entries in the vector database.")

    # Set up LangChain's conversation chain with memory
    memory = ConversationBufferWindowMemory(k=3)
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=os.getenv('OPENAI_API_KEY'))
    conversation = ConversationChain(llm=llm, memory=memory)

    # Interactive loop with conversation memory
    while True:
        user_prompt = input('Enter your query: ')

        if user_prompt.lower() == 'exit':
            break
        else:
            retrieved_data = query_vector_db(user_prompt, model, index, formatted_data)
            context = "\n".join(retrieved_data)
            final_prompt = f"System: {system}\n\nContext: {context}\n\nUser query: {user_prompt}"
            response = conversation.run(input=final_prompt)

            user_prompt_lower = user_prompt.lower()
            if any(keyword in user_prompt.lower() for keyword in plot_keywords):
                try:
                    code = extract_python_code(response)
                    # print(code)
                    image_path = execute_code_and_generate_plot(code)
                    continue
                except Exception as e:
                    print(f"Error processing plot request: {e}")

            print("Answer:", response)
