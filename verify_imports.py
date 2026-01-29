
try:
    import app
    print("Successfully imported app")
except Exception as e:
    print(f"Failed to import app: {e}")

try:
    import chatbot_core
    print("Successfully imported chatbot_core")
except Exception as e:
    print(f"Failed to import chatbot_core: {e}")

try:
    import model_loader
    print("Successfully imported model_loader")
except Exception as e:
    print(f"Failed to import model_loader: {e}")
