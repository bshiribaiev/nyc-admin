from main import FixedNYCAdminTool

# Test the improved tool
tool = FixedNYCAdminTool()

question = "What do I need for a building permit?"
result = tool.ask_question(question)

print(f"\nQuestion: {question}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.1%}")