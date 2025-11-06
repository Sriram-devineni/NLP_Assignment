
import os
from google import genai
GEMINI_API_KEY="AIzaSyAF-OsmWpu9YyImRUh1k366opCq9U9d1_o"

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words"
)
print(response.text)
