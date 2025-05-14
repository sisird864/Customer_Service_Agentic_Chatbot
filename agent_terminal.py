import pandas as pd
import sqlite3
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import requests
import os
from twilio.rest import Client
from dotenv import load_dotenv


load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolType(Enum):
    SQL_QUERY = "SQL_QUERY"
    DIRECT_RESPONSE = "DIRECT_RESPONSE"
    WHATSAPP_MESSAGE = "WHATSAPP_MESSAGE"


@dataclass
class AgentResponse:
    thinking: str
    plan: str
    action: str
    tool_used: Optional[str]
    result: Optional[str]
    response: str


@dataclass
class ConversationTurn:
    user_query: str
    agent_response: str
    context_extracted: str
    timestamp: str
   
class WhatsAppManager:
    """Manages WhatsApp messaging via Twilio"""
   
    def __init__(self):
        # These should be set as environment variables
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.whatsapp_from = os.getenv('TWILIO_WHATSAPP_FROM')  # e.g., 'whatsapp:+14155238886'
        self.business_owner_number = os.getenv('BUSINESS_OWNER_WHATSAPP')  # e.g., 'whatsapp:+1234567890'
       
        if not all([self.account_sid, self.auth_token, self.whatsapp_from, self.business_owner_number]):
            logger.warning("WhatsApp credentials not properly configured")
            self.client = None
        else:
            self.client = Client(self.account_sid, self.auth_token)
   
    def send_purchase_request(self, customer_phone: str, product_name: str, product_id: str) -> Dict[str, Any]:
        """Send purchase request to business owner"""
        if not self.client:
            return {
                "success": False,
                "error": "WhatsApp service not configured"
            }
       
        try:
            message_body = f"""🛒 NEW PURCHASE REQUEST
           
Customer Phone: {customer_phone}
Product: {product_name}
Product ID: {product_id}


Please contact the customer to complete the purchase."""
           
            message = self.client.messages.create(
                body=message_body,
                from_=self.whatsapp_from,
                to=self.business_owner_number
            )
           
            return {
                "success": True,
                "message_sid": message.sid,
                "status": message.status
            }
           
        except Exception as e:
            logger.error(f"WhatsApp send error: {e}")
            return {
                "success": False,
                "error": str(e)
            }




class DatabaseManager:
    """Manages SQLite database operations"""
   
    def __init__(self, csv_file_path: str):
        self.db_path = ":memory:"  # In-memory database
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.setup_database(csv_file_path)
        self.whatsapp_manager = WhatsAppManager()
   
    def setup_database(self, csv_file_path: str):
        """Load CSV data into SQLite database with AI-friendly column names"""
        try:
            df = pd.read_csv(csv_file_path)
           
            # Rename columns to be more AI-friendly and consistent
            column_mapping = {
                'ProductID': 'product_id',
                'ProductName': 'product_name',
                'ProductBrand': 'brand',
                'Gender': 'gender',
                'Price (INR)': 'price',
                'NumImages': 'num_images',
                'Description': 'description',
                'PrimaryColor': 'color'
            }
           
            df = df.rename(columns=column_mapping)
            df.to_sql('products', self.conn, index=False, if_exists='replace')
           
            logger.info(f"Database setup complete. Loaded {len(df)} products.")
            logger.info(f"Renamed columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
   
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
           
            if query.strip().upper().startswith('SELECT'):
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return {
                    "success": True,
                    "columns": columns,
                    "rows": rows,
                    "count": len(rows),
                    "data": [dict(zip(columns, row)) for row in rows]
                }
            else:
                self.conn.commit()
                return {
                    "success": True,
                    "message": "Query executed successfully",
                    "rows_affected": cursor.rowcount,
                    "rows": [],
                    "columns": []
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "rows": [],
                "columns": []
            }
   
    def get_schema_info(self) -> str:
        """Get database schema information with SQL best practices"""
        query = "PRAGMA table_info(products)"
        result = self.execute_query(query)
        if result["success"]:
            schema = """
DATABASE SCHEMA - 'products' table:
Columns:
- product_id (INTEGER) - Unique product identifier
- product_name (TEXT) - Full product name (e.g., 'Women Red Cotton Saree')
- brand (TEXT) - Brand name (e.g., 'Nike', 'Adidas')
- gender (TEXT) - Target gender ('Men', 'Women', 'Boys', 'Girls', 'Unisex')
- price (REAL) - Price in Indian Rupees
- num_images (INTEGER) - Number of product images
- description (TEXT) - Product description
- color (TEXT) - Primary color (e.g., 'Red', 'Blue', 'Black')


CRITICAL SQL RULES - FOLLOW EXACTLY:
1. ALWAYS use UPPER() for case-insensitive comparisons
2. ALWAYS use LIKE with % wildcards for flexible text matching
3. For cheapest items: ORDER BY price ASC LIMIT 1
4. For most expensive: ORDER BY price DESC LIMIT 1
5. Column names: product_id, product_name, brand, gender, price, num_images, description, color
6. For total count: SELECT COUNT(*) as total_count FROM products
7. Images are stored in folder with filename format: {product_id}.jpg
8. Always include product_id in SELECT queries when user asks about products
9. For product details: SELECT product_id, product_name, brand, price, color, description FROM products WHERE...


EXAMPLE QUERIES:
- Total products: SELECT COUNT(*) as total_count FROM products;
- Cheapest blue saree: SELECT * FROM products WHERE UPPER(product_name) LIKE '%SAREE%' AND UPPER(color) LIKE '%BLUE%' ORDER BY price ASC LIMIT 1;
- Count red sarees: SELECT COUNT(*) as count FROM products WHERE UPPER(product_name) LIKE '%SAREE%' AND UPPER(color) LIKE '%RED%';
- Nike shoes under 2000: SELECT * FROM products WHERE UPPER(brand) LIKE '%NIKE%' AND UPPER(product_name) LIKE '%SHOE%' AND price < 2000;
- Women's products: SELECT * FROM products WHERE UPPER(gender) LIKE '%WOMEN%';
"""
            return schema
        return "Could not retrieve schema information"


class ConversationMemory:
    """Manages conversation context and memory"""
   
    def __init__(self, max_memory: int = 2):
        self.max_memory = max_memory
        self.conversation_history: List[ConversationTurn] = []
   
    def add_turn(self, user_query: str, agent_response: str, context_extracted: str = ""):
        """Add a conversation turn to memory"""
        turn = ConversationTurn(
            user_query=user_query,
            agent_response=agent_response,
            context_extracted=context_extracted,
            timestamp=datetime.now().isoformat()
        )
       
        self.conversation_history.append(turn)
       
        # Keep only the last max_memory turns
        if len(self.conversation_history) > self.max_memory:
            self.conversation_history = self.conversation_history[-self.max_memory:]
   
    def get_context_for_llm(self) -> str:
        """Get formatted conversation context for the LLM"""
        if not self.conversation_history:
            return "No previous conversation context."
       
        context = "RECENT CONVERSATION CONTEXT:\n"
        for i, turn in enumerate(self.conversation_history, 1):
            context += f"\nTurn {i}:\n"
            context += f"User: {turn.user_query}\n"
            context += f"Assistant: {turn.agent_response}\n"
            if turn.context_extracted:
                context += f"Context: {turn.context_extracted}\n"
       
        context += "\nIMPORTANT: Use this context to understand references like 'the cheapest one', 'those products', 'that brand', etc. in the current query.\n"
        return context
   
    def extract_context_from_query(self, user_query: str) -> str:
        """Extract searchable context from user query (product types, colors, brands, etc.)"""
        context_keywords = []
       
        # Common product types
        product_types = ['saree', 'shirt', 'dress', 'jeans', 'shoe', 'kurta', 'top', 'bottom', 'jacket', 'bag']
        colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple', 'orange', 'brown']
        brands = ['nike', 'adidas', 'puma', 'levis', 'zara', 'h&m']
       
        query_lower = user_query.lower()
       
        for product in product_types:
            if product in query_lower:
                context_keywords.append(f"product_type:{product}")
       
        for color in colors:
            if color in query_lower:
                context_keywords.append(f"color:{color}")
       
        for brand in brands:
            if brand in query_lower:
                context_keywords.append(f"brand:{brand}")
       
        return ", ".join(context_keywords)
   
    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_history = []


class LlamaAgent:
    """The core agentic LLM that makes all decisions with conversation memory"""
   
    def __init__(self, db_manager: DatabaseManager, memory: ConversationMemory, lm_studio_url: str = "http://localhost:1234"):
        self.db_manager = db_manager
        self.memory = memory
        self.lm_studio_url = lm_studio_url
        self.api_endpoint = f"{lm_studio_url}/v1/chat/completions"
       
    def _call_llama(self, prompt: str) -> str:
        """
        Call Llama 3 8B via LM Studio's OpenAI-compatible API
        """
        try:
            headers = {
                "Content-Type": "application/json"
            }
           
            payload = {
                "model": "meta-llama-3.1-8b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 800,
                "stream": False
            }
           
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
           
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                raise Exception(f"LM Studio API error: {response.status_code}")
               
        except Exception as e:
            logger.error(f"Error calling LM Studio: {e}")
            raise Exception(f"Cannot connect to LM Studio: {e}")
   


    def _extract_whatsapp_details(self, response: str) -> Optional[Dict[str, str]]:
        """Extract WhatsApp message details from LLM response"""
        details = {}
       
        patterns = {
            'customer_phone': r"CUSTOMER_PHONE:\s*([^\n]+)",
            'product_name': r"PRODUCT_NAME:\s*([^\n]+)",
            'product_id': r"PRODUCT_ID:\s*([^\n]+)"
        }
       
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                details[key] = match.group(1).strip()
       
        return details if len(details) == 3 else None
   
    def _handle_whatsapp_tool(self, user_query: str, whatsapp_details: Dict) -> str:
        """Handle WhatsApp message sending"""
        # Fix: Use db_manager.whatsapp_manager instead of self.whatsapp_manager
        result = self.db_manager.whatsapp_manager.send_purchase_request(
            customer_phone=whatsapp_details['customer_phone'],
            product_name=whatsapp_details['product_name'],
            product_id=whatsapp_details['product_id']
        )
       
        if result['success']:
            return f"Great! I've sent your purchase request to our business owner. They will contact you at {whatsapp_details['customer_phone']} shortly to complete your order for {whatsapp_details['product_name']}."
        else:
            return f"I'm sorry, there was an issue sending your purchase request: {result['error']}. Please try contacting us directly."
    def _check_image_exists(self, product_id: str, image_folder: str = "images") -> str:
        """Check if product image exists and return path or placeholder"""
        if not product_id:
            return None
       
        # Common image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.webp']
       
        for ext in extensions:
            image_path = os.path.join(image_folder, f"{product_id}{ext}")
            if os.path.exists(image_path):
                return image_path
       
        return None


    def _format_product_with_image(self, product_data: Dict, image_folder: str = "images") -> str:
        """Format product information including image if available"""
        product_id = str(product_data.get('product_id', ''))
        product_name = product_data.get('product_name', 'Unknown Product')
        price = product_data.get('price', 0)
        brand = product_data.get('brand', 'Unknown Brand')
       
        # Check for image
        image_path = self._check_image_exists(product_id, image_folder)
       
        result = f"""
    🛍️ **{product_name}**
    💰 Price: ₹{price}
    🏷️ Brand: {brand}
    🆔 Product ID: {product_id}
    """
       
        if image_path:
            result += f"📸 Image: {image_path}\n"
        else:
            result += "📸 Image: Not available\n"
       
        return result






    def _get_system_prompt(self) -> str:
        """Enhanced system prompt with conversation memory capabilities"""
        return """You are an intelligent customer service agent for a fashion e-commerce store with conversation memory.


You have access to a product database and can write SQL queries when needed. You also remember the last 2 conversations to understand context and references.


DATABASE SCHEMA:
- product_id (INTEGER) - Unique identifier
- product_name (TEXT) - Full product name
- brand (TEXT) - Brand name
- gender (TEXT) - Target gender ('Men', 'Women', 'Boys', 'Girls', 'Unisex')
- price (REAL) - Price in Indian Rupees
- num_images (INTEGER) - Number of images
- description (TEXT) - Product description
- color (TEXT) - Primary color


You are an intelligent customer service agent for a fashion e-commerce store with conversation memory.


You have access to a product database and can write SQL queries when needed. You also remember the last 2 conversations to understand context and references.


You can also send WhatsApp messages to the business owner when customers want to make purchases.


- Images are stored in 'images' folder with filename format: {product_id}.jpg/.png/.jpeg/.webp
- Always include product_id in SELECT queries when showing products to users
- When displaying products, include: product name, price, brand, product_id, and image availability


DATABASE SCHEMA:
[keep existing schema info]


TOOLS AVAILABLE:
1. SQL_QUERY - For product searches and database queries
2. DIRECT_RESPONSE - For general conversation, greetings, help
3. WHATSAPP_MESSAGE - For sending purchase requests to business owner


WHEN TO USE WHATSAPP_MESSAGE:
- When user expresses intent to buy/purchase a product
- Keywords like: "I want to buy", "purchase this", "I'll take it", "can I order", "how to buy"
- Always ask for customer's phone number first if not provided
- Include product name and product_id in the message


WHATSAPP_MESSAGE FORMAT:
If user wants to buy and provides phone number, use:
TOOL_USE: WHATSAPP_MESSAGE
CUSTOMER_PHONE: [phone number]
PRODUCT_NAME: [product name]
PRODUCT_ID: [product id]


CONVERSATION FLOW FOR PURCHASES:
1. User shows buying intent -> Ask for phone number
2. User provides phone number -> Send WhatsApp message
3. Confirm message sent to business owner


CRITICAL SQL RULES - FOLLOW EXACTLY:
1. ALWAYS use UPPER() for case-insensitive comparisons
2. ALWAYS use LIKE with % wildcards for text matching
3. For finding cheapest: ORDER BY price ASC LIMIT 1
4. For finding most expensive: ORDER BY price DESC LIMIT 1
5. Use semicolon at the end of SQL queries
6. Column names are: product_id, product_name, brand, gender, price, num_images, description, color
7. For total count: SELECT COUNT(*) as total_count FROM products;


CONVERSATION MEMORY HANDLING:
- Pay attention to the conversation context provided
- When user says "the cheapest one", "those products", "that brand", etc., refer to previous conversation
- Use previous context to understand what the user is referring to
- If user asks about "the cheapest one" and previously asked about "green sarees", understand they want the cheapest green saree


WHEN TO USE SQL vs DIRECT RESPONSE:
- Use SQL_QUERY for: product searches, prices, counts, comparisons, specific product information
- Use DIRECT_RESPONSE for: greetings, general questions about your role, help requests, goodbyes


RESPONSE FORMAT (FOLLOW EXACTLY):
THINKING: [Your analysis of what the user wants, considering conversation context]
PLAN: [Step by step plan]
ACTION: [What you will do]
TOOL_USE: [SQL_QUERY or DIRECT_RESPONSE]


If TOOL_USE is SQL_QUERY, write the complete SQL query on the next line.
If TOOL_USE is DIRECT_RESPONSE, write your response on the next line.


EXAMPLES WITH CONTEXT:
Previous: User asked "Do you have green sarees?"
Current: "What is the cheapest one?"
THINKING: User previously asked about green sarees, now wants the cheapest green saree
PLAN: 1. Query products for sarees, 2. Filter by green color, 3. Order by price ascending, 4. Limit to 1 result
ACTION: Execute SQL query to find cheapest green saree
TOOL_USE: SQL_QUERY
SELECT * FROM products WHERE UPPER(product_name) LIKE '%SAREE%' AND UPPER(color) LIKE '%GREEN%' ORDER BY price ASC LIMIT 1;


Previous: User asked about "Nike shoes"
Current: "Show me the expensive ones"
THINKING: User previously asked about Nike shoes, now wants expensive Nike shoes
PLAN: 1. Query products for Nike shoes, 2. Order by price descending, 3. Show multiple results
ACTION: Execute SQL query for expensive Nike shoes
TOOL_USE: SQL_QUERY
SELECT * FROM products WHERE UPPER(brand) LIKE '%NIKE%' AND UPPER(product_name) LIKE '%SHOE%' ORDER BY price DESC LIMIT 5;


IMPORTANT: You MUST generate appropriate responses based on the query results. If you use SQL_QUERY, you will receive the database results and must create a user-friendly response from that data."""
   
    def _extract_sql_query(self, response: str) -> Optional[str]:
        """Extract SQL query from LLM response"""
        # Look for SQL query after TOOL_USE: SQL_QUERY
        sql_patterns = [
            r"TOOL_USE:\s*SQL_QUERY\s*\n(.*?)(?=\n\n|\nTHINKING|\nPLAN|\nACTION|$)",
            r"```sql\n(.*?)\n```",
            r"(SELECT.*?;)",
        ]
       
        for pattern in sql_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                # Clean up the query
                query = re.sub(r'\n+', ' ', query)
                query = re.sub(r'\s+', ' ', query)
               
                if query.upper().startswith('SELECT') and len(query) > 10:
                    if not query.endswith(';'):
                        query += ';'
                    return query
       
        return None
   
    def _extract_direct_response(self, response: str) -> Optional[str]:
        """Extract direct response from LLM response"""
        # Look for response after TOOL_USE: DIRECT_RESPONSE
        patterns = [
            r"TOOL_USE:\s*DIRECT_RESPONSE\s*\n(.*?)(?=\n\n|\nTHINKING|\nPLAN|\nACTION|$)",
            r"DIRECT_RESPONSE\s*\n(.*?)(?=\n\n|$)"
        ]
       
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
       
        return None
   
    def _determine_tool_from_response(self, response: str) -> ToolType:
        """Determine tool based on LLM's explicit choice"""
        if "TOOL_USE: WHATSAPP_MESSAGE" in response or "TOOL_USE:WHATSAPP_MESSAGE" in response:
            return ToolType.WHATSAPP_MESSAGE
        elif "TOOL_USE: SQL_QUERY" in response or "TOOL_USE:SQL_QUERY" in response:
            return ToolType.SQL_QUERY
        elif "TOOL_USE: DIRECT_RESPONSE" in response or "TOOL_USE:DIRECT_RESPONSE" in response:
            return ToolType.DIRECT_RESPONSE
       
        # If unclear, default to direct response
        logger.warning("Could not determine tool from LLM response, defaulting to DIRECT_RESPONSE")
        return ToolType.DIRECT_RESPONSE
   
    def _generate_final_response(self, user_query: str, db_result: Dict, original_llm_response: str, image_folder: str = "images") -> str:
        """Let the LLM generate the final response based on query results with image information"""
        if not db_result["success"]:
            # Let LLM handle errors
            error_prompt = f"""
    User asked: "{user_query}"
    Database error occurred: {db_result.get('error', 'Unknown error')}


    Generate a helpful response explaining the issue and suggesting alternatives.
    """
            try:
                return self._call_llama(error_prompt)
            except:
                return "I encountered an error while processing your request. Please try rephrasing your question."
       
        # Include conversation context in the response generation
        context = self.memory.get_context_for_llm()
       
        # Format products with images if we have product results
        formatted_products = ""
        if db_result.get("data") and len(db_result["data"]) > 0:
            formatted_products = "\n" + "\n".join([
                self._format_product_with_image(product, image_folder)
                for product in db_result["data"]
            ])
       
        # Let LLM generate response based on results
        response_prompt = f"""
    {context}


    Current User Query: "{user_query}"
    Database Results: {json.dumps(db_result, indent=2)}


    Formatted Products with Images:
    {formatted_products}


    Based on these database results and the conversation context above, generate a helpful, natural response to the user.
    Include the formatted product information (with images and product IDs) in your response.
    Be specific about the products found, include relevant details like names, prices, brands, product IDs, and mention if images are available.
    Make your response conversational and helpful.
    Reference the conversation context when appropriate (e.g., "Here's the cheapest green saree you asked about").
    """
       
        try:
            return self._call_llama(response_prompt)
        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            # Enhanced fallback with product formatting
            if db_result["rows"] and formatted_products:
                return f"I found {len(db_result['rows'])} result(s) for your query:\n{formatted_products}"
            elif db_result["rows"]:
                return f"I found {len(db_result['rows'])} result(s) for your query."
            else:
                return "I couldn't find any results matching your criteria."


   
    def process_query(self, user_query: str) -> AgentResponse:
        """Main agentic processing loop with conversation memory"""
       
        # Get conversation context
        conversation_context = self.memory.get_context_for_llm()
       
        # Step 1: Let LLM analyze and plan with conversation context
        thinking_prompt = f"""
You are an intelligent customer service agent for a fashion e-commerce store with conversation memory.
You have access to a product database with the following schema:


{self.db_manager.get_schema_info()}


{conversation_context}


Current User Query: "{user_query}"


Follow the exact response format from the system prompt. Analyze the request considering the conversation context and decide what tool to use.
Pay special attention to references like "the cheapest one", "those products", etc. that refer to previous conversation.
"""
       
        try:
            llm_response = self._call_llama(thinking_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return AgentResponse(
                thinking="Error: Could not connect to LLM",
                plan="Handle connection error",
                action="Return error message",
                tool_used="ERROR",
                result=str(e),
                response="I'm having trouble connecting to my AI service right now. Please try again later."
            )
       
        # Extract components from LLM response
        thinking = self._extract_section(llm_response, "THINKING:")
        plan = self._extract_section(llm_response, "PLAN:")
        action = self._extract_section(llm_response, "ACTION:")
       
        # Step 2: Execute the tool based on LLM's decision
        tool_used = self._determine_tool_from_response(llm_response)
        result = None
        final_answer = ""
       
        if tool_used == ToolType.SQL_QUERY:
            sql_query = self._extract_sql_query(llm_response)
           
            if sql_query:
                logger.info(f"Executing LLM-generated SQL: {sql_query}")
                db_result = self.db_manager.execute_query(sql_query)
                result = json.dumps(db_result, indent=2)
               
                # Let LLM generate the final response
                final_answer = self._generate_final_response(user_query, db_result, llm_response, "images")
            else:
                logger.error("LLM failed to generate valid SQL query")
                final_answer = "I had trouble generating a proper database query for your request. Could you please rephrase your question?"
                result = "Error: No valid SQL query generated by LLM"
       
        elif tool_used == ToolType.DIRECT_RESPONSE:
            # Extract the direct response from LLM
            direct_response = self._extract_direct_response(llm_response)
            if direct_response:
                final_answer = direct_response
            else:
                # If we can't extract it, use a generic response
                final_answer = "I'd be happy to help you with your fashion needs! Please let me know what specific products you're looking for."
            result = "Direct response from LLM"
       
        elif tool_used == ToolType.WHATSAPP_MESSAGE:
            whatsapp_details = self._extract_whatsapp_details(llm_response)
           
            if whatsapp_details:
                logger.info(f"Sending WhatsApp message for purchase: {whatsapp_details}")
                final_answer = self._handle_whatsapp_tool(user_query, whatsapp_details)
                result = f"WhatsApp sent to business owner for: {whatsapp_details['product_name']}"
            else:
                logger.error("LLM failed to provide complete WhatsApp details")
                final_answer = "I need your phone number and the specific product you want to purchase. Could you please provide both?"
                result = "Error: Incomplete WhatsApp message details from LLM"


       
        return AgentResponse(
            thinking=thinking,
            plan=plan,
            action=action,
            tool_used=tool_used.value,
            result=result,
            response=final_answer
        )
   
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from LLM response"""
        pattern = f"{section_name}\\s*(.*?)(?=\\n[A-Z]+:|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""


class AgenticRAGSystem:
    """Main system orchestrator with conversation memory"""
   
    def __init__(self, csv_file_path: str, lm_studio_url: str = "http://localhost:1234", memory_turns: int = 2, image_folder: str = "images"):
        self.db_manager = DatabaseManager(csv_file_path)
        self.memory = ConversationMemory(max_memory=memory_turns)
        self.agent = LlamaAgent(self.db_manager, self.memory, lm_studio_url)
        self.image_folder = image_folder
        self.session_history = []
       
        # Test LM Studio connection
        self._test_connection()
        self._test_whatsapp_config()
        self._check_image_folder()
   
    def _check_image_folder(self):
        """Check if image folder exists"""
        if os.path.exists(self.image_folder):
            image_count = len([f for f in os.listdir(self.image_folder)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            logger.info(f"✅ Image folder '{self.image_folder}' found with {image_count} images!")
        else:
            logger.warning(f"⚠️  Image folder '{self.image_folder}' not found - images will not be displayed")
            logger.info(f"Create folder '{self.image_folder}' and add product images named as ProductID.jpg")




    def _test_whatsapp_config(self):
        """Test WhatsApp configuration"""
        if self.db_manager.whatsapp_manager.client:
            logger.info("✅ WhatsApp configuration loaded successfully!")
        else:
            logger.warning("⚠️  WhatsApp not configured - purchase requests will not work")
            logger.info("Set environment variables: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM, BUSINESS_OWNER_WHATSAPP")


   
    def _test_connection(self):
        """Test connection to LM Studio"""
        try:
            response = requests.get(f"{self.agent.lm_studio_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                logger.info("✅ LM Studio connection successful!")
                logger.info(f"Available models: {[model['id'] for model in models.get('data', [])]}")
            else:
                logger.warning("⚠️  LM Studio connection issue - check if server is running")
                raise Exception("LM Studio not responding properly")
        except Exception as e:
            logger.error(f"❌ Could not connect to LM Studio: {e}")
            logger.error("Make sure LM Studio is running and serving on the correct port")
            raise Exception(f"LM Studio connection failed: {e}")
   
    def chat(self, user_input: str) -> Dict[str, Any]:
        """Process user input and return comprehensive response with memory"""
       
        timestamp = datetime.now().isoformat()
        logger.info(f"Processing query: {user_input}")
       
        # Let the LLM agent handle everything
        agent_response = self.agent.process_query(user_input)
       
        # Extract context from the current query
        context_extracted = self.memory.extract_context_from_query(user_input)
       
        # Add this conversation turn to memory
        self.memory.add_turn(
            user_query=user_input,
            agent_response=agent_response.response,
            context_extracted=context_extracted
        )
       
        # Log the agentic process
        logger.info("=== AGENTIC PROCESS WITH MEMORY ===")
        logger.info(f"THINKING: {agent_response.thinking}")
        logger.info(f"PLAN: {agent_response.plan}")
        logger.info(f"ACTION: {agent_response.action}")
        logger.info(f"TOOL_USED: {agent_response.tool_used}")
        logger.info(f"CONTEXT_EXTRACTED: {context_extracted}")
        if agent_response.result:
            logger.info(f"RESULT: {agent_response.result[:200]}...")
       
        # Store in session history
        session_entry = {
            "timestamp": timestamp,
            "user_query": user_input,
            "agent_response": agent_response.__dict__,
            "context_extracted": context_extracted,
            "conversation_memory": [turn.__dict__ for turn in self.memory.conversation_history]
        }
        self.session_history.append(session_entry)
       
        return {
            "response": agent_response.response,
            "debug_info": {
                "thinking": agent_response.thinking,
                "plan": agent_response.plan,
                "action": agent_response.action,
                "tool_used": agent_response.tool_used,
                "result": agent_response.result,
                "context_extracted": context_extracted,
                "memory_turns": len(self.memory.conversation_history)
            },
            "timestamp": timestamp
        }
   
    def get_session_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.session_history
   
    def get_conversation_memory(self) -> List[ConversationTurn]:
        """Get current conversation memory"""
        return self.memory.conversation_history
   
    def reset_session(self):
        """Reset conversation history and memory"""
        self.session_history = []
        self.memory.clear_memory()
        logger.info("Session history and conversation memory reset")


# Example usage and testing
def main():
    """Example usage of the Agentic RAG system with conversation memory"""
   
    # Initialize the system
    csv_file = "products.csv"
    lm_studio_url = "http://localhost:1234"
   
    print("🤖 Agentic Customer Service AI with Conversation Memory")
    print("=" * 60)
    print("⚡ Connecting to LM Studio...")
   
    try:
        rag_system = AgenticRAGSystem(csv_file, lm_studio_url, memory_turns=2, image_folder="images")
        print("🚀 System Ready! The LLM remembers the last 2 conversations.")
        print("Try: 'Do you have green sarees?' then 'What is the cheapest one?'")
        print("=" * 60)
       
        # Interactive mode
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
               
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
               
                if user_input.lower() == 'memory':
                    print("\n🧠 Current Conversation Memory:")
                    memory = rag_system.get_conversation_memory()
                    if not memory:
                        print("   No previous conversations in memory.")
                    else:
                        for i, turn in enumerate(memory, 1):
                            print(f"   Turn {i}: {turn.user_query} -> {turn.agent_response[:50]}...")
                    continue
               
                if user_input.lower() == 'reset':
                    rag_system.reset_session()
                    print("🔄 Session and memory reset!")
                    continue
               
                if not user_input:
                    continue
                   
                print("🧠 LLM is thinking with conversation context...")
                result = rag_system.chat(user_input)
                print(f"🤖 Assistant: {result['response']}")
               
                # Show debug info to see the agentic process
                print(f"\n   🔍 Agentic Process:")
                print(f"   💭 Thinking: {result['debug_info']['thinking']}")
                print(f"   📋 Plan: {result['debug_info']['plan']}")
                print(f"   🔧 Tool Used: {result['debug_info']['tool_used']}")
                print(f"   🧠 Memory Turns: {result['debug_info']['memory_turns']}")
               
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
               
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Make sure LM Studio is running with a model loaded!")


if __name__ == "__main__":
    main()

