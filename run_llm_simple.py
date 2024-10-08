import os
from utils.BiographyVectorStore import retrieve_vector_store
from langchain_huggingface import HuggingFaceEmbeddings
from environment.llm_prisoners_dilemma_game import PrisonersDilemmaGame

class PrisonersDilemmaGameRunner:
    def __init__(self, agent1_name, agent2_name, vector_store_dir, num_rounds=5):
        """Initializes the game runner with the agent names, vector store directory, and number of rounds."""
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.vector_store_dir = vector_store_dir
        self.num_rounds = num_rounds
        self.vectorstore = self.initialize_vector_store()
        
    def initialize_vector_store(self):
        """Initializes and retrieves the vector store from the directory."""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        return retrieve_vector_store(self.vector_store_dir, embeddings)
    
    def run_game(self):
        """Runs the Prisoner's Dilemma game between the two agents."""
        game = PrisonersDilemmaGame(
            num_rounds=self.num_rounds,
            agent1_name=self.agent1_name,
            agent2_name=self.agent2_name,
            vectorstore=self.vectorstore
        )
        game.play()
        
        # Get and print the final scores
        final_scores = game.get_scores()
        print("Final Scores:")
        print(f"{self.agent1_name}: {final_scores[0][-1]}")
        print(f"{self.agent2_name}: {final_scores[1][-1]}")

if __name__ == "__main__":
    # Define agent names and vector store directory
    agent1_name = "Pope Francis"
    agent2_name = "Benjamin Netanyahu"
    vector_store_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/vector_store')

    # Create the game runner and run the game
    game_runner = PrisonersDilemmaGameRunner(agent1_name, agent2_name, vector_store_dir, num_rounds=5)
    game_runner.run_game()

