from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()

class GameAgent:
    def __init__(self, agent_name: str, vectorstore, system_message: str):
        self.agent_name = agent_name
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever()
        
        # Ensure that the OpenAI API key is loaded
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is missing. Make sure it is set in the .env file.")

        
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=openai_key)
        self.memory = MemorySaver()
        self.tools = [TavilySearchResults(max_results=3)] 
        self.agent_executor = create_react_agent(self.model, self.tools, checkpointer=self.memory)
        self.chat_history = [SystemMessage(content=system_message)]
        self.config = {"configurable": {"thread_id": "abc123"}}

    def add_prompt_message(self, content: str):
        self.chat_history.append(HumanMessage(content=content))

    def get_agent_response(self, prompt: str, opponent_name: str):
        self.add_prompt_message(prompt)
        reasoning = ""  # Variable to store reasoning
        print(f"Prompt to {self.agent_name}:")
        print(prompt)
        for chunk in self.agent_executor.stream({"messages": self.chat_history}, self.config):
            reasoning_chunk = chunk['agent']['messages'][0].content.strip()
            reasoning += reasoning_chunk  # Accumulate reasoning chunks
        response = reasoning[0].upper()  # Assuming the first character is the decision
        self.chat_history.pop()
        return response, reasoning

    def retrieve_information(self, name: str) -> str:
        # Use the vector store to retrieve information about the person
        try:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 1, "filter": {"name": name}},
            )
            results = retriever.invoke("")
        except Exception as e:
            print(f"Error during retrieval: {e}")
            results = []
        if results:
            full_text = results[0].page_content
            summary = full_text[:10000]  # Limit the text to 10,000 characters
            return summary
        else:
            return f"No background information available for {name}."
