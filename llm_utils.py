async def parse_query_with_llm_async(client, model, query):
    """Asynchronously parses a query using the LLM."""
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
    res = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

async def get_decision_llm_async(client, model, structured_query, clauses):
    """Asynchronously gets a decision from the LLM."""
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
    res = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# NEW, SIMPLER FUNCTION FOR DIRECT Q&A
async def get_direct_answer_async(client, model, question, clauses):
    """
    Asynchronously gets a direct, factual answer from the LLM based on provided text.
    """
    prompt = f'''
You are a helpful insurance policy assistant.
Based *only* on the provided "Relevant Clauses" from the policy document, provide a concise and direct answer to the "Question".
Do not make decisions or assumptions. Stick strictly to the text provided.

Relevant Clauses:
"""
{clauses}
"""

Question:
"{question}"

Answer:
'''
    res = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content