def parse_query_with_llm(client, model, query):
    prompt = f'''
Extract structured information from the following insurance query:
"{query}"
Return a JSON object with keys:
- age
- gender
- procedure
- location
- policy_duration
'''
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

def get_decision_llm(client, model, structured_query, clauses):
    prompt = f'''
You are a claim assistant.
Structured Query:
{structured_query}
Relevant Clauses:
{clauses}
Decide whether the claim is approved or rejected, estimate payout if applicable, and explain your reasoning very briefly referencing specific clauses.
Respond in JSON:
{{
"decision": "...",
"amount": "...",
"justification": "..."
}}
'''
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content
