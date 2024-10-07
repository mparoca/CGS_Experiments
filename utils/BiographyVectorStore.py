"""
BiographyVectorStore Module

This module provides functionality to scrape biographical data from Wikipedia,
save it locally, and create a vector store using the Chroma library. The vector
store is persisted to disk and can be retrieved later without re-running the
data processing steps.

Classes:
    BiographyVectorStore: A class to manage the scraping, saving, and vectorization
    of Wikipedia biographies.

Functions:
    retrieve_vector_store(vector_store_dir, embeddings): Retrieves the vector store
    from the specified directory using the provided embedding function.

Usage:
    # Define directories
    scraped_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/scraped')
    vector_store_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/vector_store')

    # Initialize the BiographyVectorStore
    biography_store = BiographyVectorStore(scraped_data_dir, vector_store_dir)

    # Run the process with names (only needed once to create the store)
    names = ["Stev Jobs", "Jane Goodall", "Barack Obama"]
    biography_store.run(names)

    # Retrieve the vector store without re-running the process
    vectorstore = retrieve_vector_store(vector_store_dir, biography_store.embeddings)
"""

import os
import wikipedia
import json
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import uuid  # Import the uuid module for generating unique IDs

class BiographyVectorStore:
    def __init__(self, scraped_data_dir, vector_store_dir):
        self.scraped_data_dir = scraped_data_dir
        self.vector_store_dir = vector_store_dir

        # Ensure directories exist
        os.makedirs(self.scraped_data_dir, exist_ok=True)
        os.makedirs(self.vector_store_dir, exist_ok=True)

        # Initialize embeddings
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

    def get_biography(self, name):
        """Fetches the biography of a person from Wikipedia."""
        try:
            content = wikipedia.page(name).content
            print(f"Successfully fetched biography for {name}.")
            # Optionally, print the first 100 characters of the biography
            print(f"Snippet: {content[:100]}...\n")
            return content
        except Exception as e:
            print(f"Error fetching page for {name}: {e}")
            return ""

    def save_biography(self, name, content):
        """Saves the biography content to a file."""
        file_path = os.path.join(self.scraped_data_dir, f"{name.replace(' ', '_')}.txt")
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

    def load_biographies(self, names):
        """Loads biographies from Wikipedia and saves them locally."""
        texts = []
        for name in names:
            bio = self.get_biography(name)
            if bio:
                self.save_biography(name, bio)
                texts.append((name, bio))
        return texts

    def create_document_store(self, texts):
        """Creates a document store from the texts and persists it."""
        documents = []
        for name, text in texts:
            # Generate a unique ID for each person
            person_id = str(uuid.uuid4())
            # Create a document with metadata including the name and ID
            doc = Document(page_content=text, metadata={'name': name, 'id': person_id})
            documents.append(doc)
        
        return Chroma.from_documents(documents=documents, embedding=self.embeddings, persist_directory=self.vector_store_dir)

    def run(self, names):
        """Executes the process of loading biographies and creating a document store."""
        texts = self.load_biographies(names)
        vectorstore = self.create_document_store(texts)
        print("Vector store created and saved successfully.")

def retrieve_vector_store(vector_store_dir, embeddings):
    """Retrieves the vector store with the embedding function from the persisted directory."""
    return Chroma(persist_directory=vector_store_dir, embedding_function=embeddings)


# Example usage

# # Test to see how Wikipedia interprets name input. Must test this before adding to the list of names to ensure the correct name is used. 
# # Steve Jobs is actually recognized as Stev Jobs so be careful!
# print(wikipedia.suggest("Jane Goodall")) # This returns Jane Goodal

# names = ["Stev Jobs", "Jane Goodall", "Barack Obama"]
# scraped_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/scraped')
# vector_store_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/vector_store')

# biography_store = BiographyVectorStore(scraped_data_dir, vector_store_dir)
# biography_store.run(names)

# # Test to see if the vector store can be retrieved
# vectorstore = retrieve_vector_store(vector_store_dir, biography_store.embeddings)

# # Test to see if vector store is working
# print(vectorstore.similarity_search("chimpanzees"))