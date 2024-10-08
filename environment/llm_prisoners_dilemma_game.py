from agents.llm_game_agent import GameAgent
from agents.llm_prompts import get_system_prompt, get_task_prompt
from langchain_core.messages import SystemMessage

class PrisonersDilemmaGame:
    def __init__(self, num_rounds, agent1_name, agent2_name, vectorstore):
        self.vectorstore = vectorstore
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.agent1 = GameAgent(agent_name=agent1_name, vectorstore=vectorstore, system_message="")
        self.agent2 = GameAgent(agent_name=agent2_name, vectorstore=vectorstore, system_message="")
        self.num_rounds = num_rounds
        self.history1 = []
        self.history2 = []
        self.score1 = 0
        self.score2 = 0
        self.scores_round1 = []
        self.scores_round2 = []
        self.reasoning1 = []
        self.reasoning2 = []

        # Retrieve background information
        agent1_info = self.agent1.retrieve_information(agent1_name)
        agent2_info = self.agent2.retrieve_information(agent2_name)

        # Create system messages
        system_message1 = get_system_prompt().format(
            agent_name=self.agent1_name,
            opponent_name=self.agent2_name,
            agent_info=agent1_info,
            opponent_info=agent2_info
        )

        system_message2 = get_system_prompt().format(
            agent_name=self.agent2_name,
            opponent_name=self.agent1_name,
            agent_info=agent2_info,
            opponent_info=agent1_info
        )

        # Set system messages
        self.agent1.chat_history = [SystemMessage(content=system_message1)]
        self.agent2.chat_history = [SystemMessage(content=system_message2)]

    def play(self):
        for round_number in range(1, self.num_rounds + 1):
            print(f"Round {round_number}")

            # Generate prompts for both agents
            agent1_prompt = get_task_prompt(self.history1, self.score1, self.history2, self.score2)
            agent2_prompt = get_task_prompt(self.history2, self.score2, self.history1, self.score1)

            # Get decisions and reasoning from both agents
            decision1, reasoning1 = self.agent1.get_agent_response(agent1_prompt, self.agent2_name)
            decision2, reasoning2 = self.agent2.get_agent_response(agent2_prompt, self.agent1_name)
            
            # Store reasoning
            self.reasoning1.append(reasoning1)
            self.reasoning2.append(reasoning2)

            # Update histories
            self.history1.append(decision1)
            self.history2.append(decision2)

            # Update scores
            self.update_scores(decision1, decision2)

            # Store scores for each round
            self.scores_round1.append(self.score1)
            self.scores_round2.append(self.score2)

            # Print the results of the round
            print(f"{self.agent1_name} Decision: {decision1}, {self.agent2_name} Decision: {decision2}")
            print(f"Scores -> {self.agent1_name}: {self.score1}, {self.agent2_name}: {self.score2}\n")
            print(f"{self.agent1_name} Reasoning: {reasoning1}")
            print(f"{self.agent2_name} Reasoning: {reasoning2}")

    def update_scores(self, decision1, decision2):
        payoff_matrix = {
            ('C', 'C'): (3, 3),
            ('C', 'D'): (0, 5),
            ('D', 'C'): (5, 0),
            ('D', 'D'): (1, 1)
        }
        result = payoff_matrix.get((decision1, decision2), (0, 0))
        self.score1 += result[0]
        self.score2 += result[1]

    def get_scores(self):
        """Returns the accumulated scores per round for both agents."""
        return self.scores_round1, self.scores_round2
