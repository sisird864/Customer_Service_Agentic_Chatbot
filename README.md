# **AI-Powered Product Inquiry Chatbot**

This project adds an AI-driven chat assistant to an existing e‑commerce website. Visitors can ask questions about products (by name, ID, color, price, etc.), and the chatbot responds instantly by querying your product database and returning formatted results, including images when available.

---

## **🔄 Usage Modes**

This project supports two modes of interaction:

1. **Terminal Chat Mode**: Run the Flask server and interact via a terminal-based chat client. Ideal for testing and quick demos.  
2. **API Integration Mode**: Use the `POST /api/chat` endpoint to integrate the chatbot into any website or frontend. Hook up your JavaScript chat widget to consume the API for real-time customer support.

---

## **🚀 Features**

* **AI-Powered Responses**: Natural-language replies via an RAG-enabled LLM agent.  
* **Product Lookup**: Dynamic SQL queries against a local product CSV → SQLite database.  
* **Image Rendering**: Inline product images served from a local `images/` folder. (Will not work on the terminal version)

---

## **🖥️ Prerequisites**

* Python 3.10+  
* A virtual environment (recommended)  
* Your own product CSV (`products.csv` or similarly named)  
* LM Studio: Running locally to serve the Llama 3.1-8B model via its OpenAI-compatible API (http://localhost:1234). If you prefer, you can replace this with any other compatible LLM endpoint by updating the lm\_studio\_url in app.py.

---

## **📁 Project Structure**

      Customer\_Service\_Agentic\_Chatbot/  
     ├── agentic\_rag.py        \# Agentic RAG system setup & logic  
     ├── products.csv           \# Your product data file (rename as needed)  
     ├── requirements.txt     \# Python dependencies  
     ├── images/  
      │   └──  \<product\_id\>.jpg/png  
     ├── static/  
     │   └── images/            \# \*\*CREATE THIS FOLDER\*\*  
     │       └── \<product\_id\>.jpg/png  
     └── templates/             \# Only needed if using the agent\_api to connect to a website  
           └── index.html         \# Your existing site HTML with chat widget added

## **⚙️ Installation**

1. **Clone the repository**  
   git clone https://github.com/sisird864/Customer\_Service\_Agentic\_Chatbot.git  
   cd Customer\_Service\_Agentic\_Chatbot  
   

       2\. **Create & activate a virtual environment**  
	python \-m venv venv  
source venv/bin/activate       \# macOS/Linux  
venv\\Scripts\\activate.bat    \# Windows

       3\. **Install dependencies**  
	pip install \-r requirements.txt

       4\. **Prepare your product data**

* Place your CSV file in the project root.  
* **Rename** the reference in the agent Python file to match your CSV filename (default: `products.csv`).

       5\. **Create & populate images folder**

	mkdir \-p static/images

* Save each product image in `static/images/` using its **product\_id** as the filename (e.g., `10017413.jpg`, `20000004.png`).

## **📡 API Integration in Your Own Website**

Here’s a minimal example of how to integrate the `/api/chat` endpoint into any existing site:

**1\. Include a chat widget trigger** anywhere in your HTML:

\<button id="chatButton"\>💬 Chat\</button\>

\<div id="chatWindow" style="display:none;"\>...\</div\>

**2\. Send user messages to the API**:

async function sendMessage(message) {

  const res \= await fetch('/api/chat', {

    method: 'POST',

    headers: { 'Content-Type': 'application/json' },

    body: JSON.stringify({ message })

  });

  const data \= await res.json();

  return data.response;

}

**3\. Render responses** in your chat UI, handling `IMAGE_URL:/static/images/...` and JSON `PRODUCT_IMAGES` blocks to display images.

Adjust selectors, styling, and HTML structure to match your site’s look and feel. The API remains the same—just POST `{ message: "user text" }` to `/api/chat` and use the returned string to display the bot’s reply.

## **🚀 Running the App**

    export FLASK\_APP=app.py            \# macOS/Linux  
    set FLASK\_APP=app.py               \# Windows  
    flask run                          \# launches on http://127.0.0.1:5000

Visiting your site will show the chat widget. Ask questions like:

* “Do you have this saree in green?”  
* “What’s the price of product ID 10140435?”

## **🔧 Customization**

* **CSV Schema**: Ensure your CSV has columns: `product_id`, `product_name`, `brand`, `price`, `color`, `description`, `image_url` (optional).  
* **Chat Behavior**: Modify prompt logic in the Python file to adjust conversation style or supported queries.  
* **Model Endpoint**: Update the LLM API URL & credentials in your environment or `.env` file.

---

## **🙋‍♀️ Support**

If you run into issues, please open an issue or contact the project maintainer.

