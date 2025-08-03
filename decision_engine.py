import ollama

def get_decision_llm(structured_query_json, retrieved_clauses):
    prompt = f"""
You are a claim assessment assistant.

Given this structured query:
{structured_query_json}

And the following relevant clauses from the policy:
{retrieved_clauses}

Decide whether the claim is approved or rejected, estimate payout if applicable, and explain your reasoning with reference to specific clauses.

Respond in this JSON format:
{{
  "decision": "approved or rejected",
  "amount": "estimated payout or null",
  "justification": "explanation with reference to clauses"
}}
"""
    res = ollama.chat(model="phi3", messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"]
