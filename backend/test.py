from main import RealTimeNYCAdminTool

print("Testing NYC Admin Code Tool...")

# Create the tool
tool = RealTimeNYCAdminTool()
print("✅ Tool created successfully!")

# Ask a simple question
question = "What do I need for a building permit?"
print(f"\n🤔 Asking: {question}")

try:
    result = tool.ask_question(question)
    print(f"\n📝 Answer: {result['answer']}")
    print(f"\n📚 Sources: {len(result['sources'])} sections found")
    
    for source in result['sources']:
        print(f"  - Section {source['section_id']}: {source['title']}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Don't worry! This is normal - we'll debug together.")