from src.agent_engine import execute_twin_agent

# Use a static mock session ID
session = "test_evaluation_user_101"

print("--- Turn 1 ---")
resp1 = execute_twin_agent(session, "Hi Professor Andrew! My name is Gargi. I am a student at DTU studying Electronics and Communication Engineering. I am building a gesture-controlled drone project using CNNs!")
print(f"Twin Response:\n{resp1}\n")

print("--- Turn 2 (Testing Context Short-Term Recall) ---")
resp2 = execute_twin_agent(session, "What branch did I say I am pursuing?")
print(f"Twin Response:\n{resp2}\n")