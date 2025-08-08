import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists in the project root
load_dotenv()

# Get API key from environment variable
# This will now find the key loaded from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate the key is present
# This check remains crucial to prevent running with a missing key
if not GROQ_API_KEY:
    raise ValueError("Please set your GROQ_API_KEY in a .env file or as a system environment variable.")