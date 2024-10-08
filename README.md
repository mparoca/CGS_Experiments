# LLM-based Prisoner's Dilemma Game

This project simulates a Prisoner's Dilemma game between agents embodying real-world personas. These agents make decisions based on their Wikipedia biographies, stored in a vector store using Chroma, and powered by GPT-4 for reasoning.

## **1. Setup and Installation**

### **1.1 Setting up the Python Environment**

To set up a Conda environment and install the necessary libraries, follow these steps:

```bash
# Create a new Conda environment with Python 3.9
conda create -n pd_simulation python=3.9

# Activate the Conda environment
conda activate pd_simulation

# Upgrade pip
pip install --upgrade pip

# Install necessary Python libraries
pip install langchain
pip install transformers
pip install langgraph
pip install wikipedia
pip install -U langchain-community
pip install chromadb
pip install langchain-openai
pip install python-dotenv
pip install -U langchain-community tavily-python
pip install -U langchain-huggingface
pip install -U langchain-chroma
```

### **1.2 API Keys Setup**

You will need API keys for OpenAI and Tavily to run this project. Here's how you can obtain them:

1. **OpenAI API Key**:
   - Visit [OpenAI API](https://beta.openai.com/signup/).
   - Sign up for an account or log in.
   - Go to your API settings and copy your API key.
   
2. **Tavily API Key**:
   - Visit [Tavily API](https://tavily.com) and sign up for an account.
   - Generate an API key from your account settings.

### **1.3 Creating the `.env` File**

Create a `.env` file in the root of your project and add your OpenAI and Tavily API keys like this:

```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Make sure to replace `your_openai_api_key_here` and `your_tavily_api_key_here` with your actual API keys.

---

## **2. Project Structure**

The codebase is organized into the following folders:

- **`agents/`**: Contains the `GameAgent` class and the LLM prompt templates.
  - `llm_game_agent.py`: Handles the LLM-powered agent logic.
  - `llm_prompts.py`: Contains prompt templates for generating agent decisions.
  
- **`environment/`**: Contains the game logic for the Prisoner's Dilemma simulation.
  - `llm_prisoners_dilemma_game.py`: Manages game flow, rounds, and agent interactions.
  
- **`utils/`**: Contains utility scripts for scraping Wikipedia biographies and managing the vector store.
  - `BiographyVectorStore.py`: Scrapes Wikipedia and builds the vector store using Chroma.
  
- **`data/`**: Contains the pre-scraped data and vector store.
  - `scraped/`: Stores text files of biographies.
  - `vector_store/`: Stores the vector database.
  
- **`run_llm_simple.py`**: The main script to run the game simulation.

---

## **3. Data and Utility Script**

### **3.1 Using Pre-existing Data**

The `data/` folder already contains pre-scraped Wikipedia biographies and a vector store. If you want to use the existing data, you don’t need to re-run the scraping process.

### **3.2 Recreating the Vector Store**

To scrape new Wikipedia biographies and recreate the vector store, you can run the `BiographyVectorStore.py` script. Here's how:

```bash
python utils/BiographyVectorStore.py
```

This will scrape biographies, save them locally, and create a new vector store with Chroma.

---

## **4. Running the Main Simulation**

Once all dependencies are installed, and the `.env` file is set up, you can run the Prisoner's Dilemma simulation.

```bash
python run_llm_simple.py
```

This will run a 5-round simulation between two agents—currently Pope Francis and Benjamin Netanyahu. The agents will make decisions based on their biographies, and the reasoning will be generated using GPT-4.

---

## **5. Customization**

### **Changing Agents**

If you want to simulate the game with different agents, edit the `run_llm_simple.py` script and change the `agent1_name` and `agent2_name` variables. Make sure their biographies are available in the vector store or scrape new ones using the utility script.

### **Changing Number of Rounds**

You can modify the `num_rounds` parameter in the `run_llm_simple.py` script to adjust the number of rounds in the game.

---

## **6. Acknowledgments**

This project uses APIs and libraries provided by:

- [OpenAI](https://openai.com) for GPT-4.
- [Tavily](https://tavily.com) for search and interaction tools.
- [Wikipedia](https://www.wikipedia.org/) for scraping biographies.
- [HuggingFace](https://huggingface.co/) for embeddings.
- [Chroma](https://www.trychroma.com/) for managing the vector store.
- [LangChain](https://langchain.com/) for the framework.

---