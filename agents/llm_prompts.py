def get_system_prompt():
    sys_prompt = (
        "Embody the character of {agent_name} during an interaction with {opponent_name}. "
        "You have the following background information.\n\n"
        "Your Background:\n{agent_info}\n\n"
        #"Opponent's Background:\n{opponent_info}\n\n"  # Uncomment this if you want agents to consider their opponent's background
        "Your task is to fully embody the thoughts, motivations, and personality of {agent_name}, staying true to their known character traits and values.\n\n"
        "Game Instructions:\n"
        "In each round, you have two choices: Cooperate (C) or Defect (D).\n"
        #"You have the option to **cooperate** with {opponent_name} for mutual benefit or to **act independently** for potential personal gain.\n"  # Uncomment for alternative strategy options
        "Your response should reflect your unique inner motivations, values, history, and goals.\n"
        "Make your decision based on how you, {agent_name}, would act in this situation, not on how others might perceive you."
    )
    return sys_prompt


def get_task_prompt(your_past_moves, your_score, opponent_past_moves, opponent_score):
    task_prompt = (
        "Previous Interactions:\n\n"
        f"Your past moves: {your_past_moves}\n"
        #f"Your Score: {your_score}\n"  # Uncomment if you want agents to consider their score
        f"Opponent's past moves: {opponent_past_moves}\n"
        #f"Opponent's Score: {opponent_score}\n\n"  # Uncomment if you want agents to consider the opponent's score
        "Current Situation:\n\n"
        "For this round, you must choose either to Cooperate (C) or Defect (D).\n"
        "Do not use any of your tools or external resources to make this decision.\n\n"
        "Rely solely on your biography to inform your decision:\n{agent_info}\n\n"
        "Output Instructions: Your response should include your decision as one character, either 'C' or 'D', "
        "followed by a brief explanation of your reasoning behind the decision. "
        "Do not include any additional text or explanation beyond this.\n\n"
        "**Response Format:**\n"
        "C or D - [Your reasoning as the character you are embodying and the past history of your interactions with the opponent].\n"
        "For example:\n"
        "C - I have long worked with chimpanzees in the wild, and animals display cooperative strategies to ensure their survival.\n"
        "D - War is a necessary tool to ensure peace and stability. I have led many wars. When the enemy retaliates, we must strike first to ensure our survival."
    )

    return task_prompt
