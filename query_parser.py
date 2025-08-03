import ollama

def parse_query_with_llm(query):
    prompt = f"""
Extract structured information from the following insurance query:

"{query}"

Return a JSON object with the following keys:
- age
- gender
- procedure
- location
- policy_duration
"""
    res = ollama.chat(model='phi3', messages=[{'role': 'user', 'content': prompt}])
    return res['message']['content']
