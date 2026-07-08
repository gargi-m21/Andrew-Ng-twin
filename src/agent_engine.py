import os
import json
import threading
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from src.config import ANDREW_NG_SYSTEM_PROMPT
from src.ingestion import get_retriever
from src.memory_manager import DualLayerMemoryManager

load_dotenv()
memory_manager = DualLayerMemoryManager()

def format_docs(docs):
    """Combines retrieved document fragments cleanly into a context string."""
    return "\n\n".join(f"Source Document: {os.path.basename(doc.metadata.get('source', 'Unknown'))} (Page {doc.metadata.get('page', 'N/A')})\n{doc.page_content}" for doc in docs)

def run_reflection_async(llm, reflection_prompt, session_id):
    """Asynchronous worker to process long-term memory updates without blocking the UI."""
    try:
        updated_profile = llm.invoke(reflection_prompt).content
        memory_manager.update_long_term_summary(session_id, updated_profile)
    except Exception as e:
        print(f"Background memory reflection error: {e}")

def execute_twin_agent(session_id: str, user_query: str):
    """Executes the twin agent loop, utilizing two-stage hallucination validation logic."""
    short_term_history = memory_manager.get_short_term(session_id)
    long_term_profile = memory_manager.get_long_term_summary(session_id)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2)
    
    # 1. Pull Grounded Content from your 935 local database chunks
    retriever = get_retriever()
    retrieved_docs = retriever.invoke(user_query)
    context_str = format_docs(retrieved_docs)
    
    # 2. Build the Core LCEL Generation Pipeline
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", ANDREW_NG_SYSTEM_PROMPT.format(long_term_memory=long_term_profile)),
        MessagesPlaceholder(variable_name="history"),
        ("system", "Grounding Context from notes and research papers:\n\n{context}"),
        ("human", "{input}")
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    initial_answer = chain.invoke({
        "history": short_term_history,
        "context": context_str,
        "input": user_query
    })
    
    # 3. Two-Stage Validation: Incorporating JSON verification loop
    verifier_prompt = f"""
    You are a Hallucination Verifier for a RAG system simulating Professor Andrew Ng.
    Analyze the generated answer against the retrieved context to ensure absolute structural alignment.

    Retrieved Context:
    {context_str}

    Generated Answer:
    {initial_answer}

    Tasks:
    1. Check for any technical assertions or claims not supported by the context.
    2. Rewrite the answer if any hallucinations are present while maintaining Andrew's supportive first-person persona.
    
    Respond strictly with a JSON object containing keys:
    - "is_hallucination": boolean
    - "explanation": string
    - "final_answer": string
    """
    
    # Instantiate an explicit JSON-mode LLM instance
    json_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=api_key, 
        generation_config={"response_mime_type": "application/json"}
    )
    
    try:
        verification_response = json_llm.invoke(verifier_prompt).content
        result = json.loads(verification_response)
        is_hallucination = result.get("is_hallucination", False)
        explanation = result.get("explanation", "")
        final_answer = result.get("final_answer", initial_answer)
    except Exception as e:
        is_hallucination = False
        explanation = f"Verification fallback active: {str(e)}"
        final_answer = initial_answer

    # 4. Save validated turn to short term memory history
    memory_manager.add_short_term(session_id, user_query, final_answer)
    
    # 5. Background Threading: Async long-term insight extraction
    reflection_prompt = f"""
    Update the student's structural data log. 
    Current State: {long_term_profile}
    Latest Turn: Student asked "{user_query}" -> Twin responded "{final_answer}"
    Compile permanent details briefly. Do not invent metrics.
    """
    threading.Thread(
        target=run_reflection_async, 
        args=(llm, reflection_prompt, session_id), 
        daemon=True
    ).start()
        
    hallucination_info = {
        "is_hallucination": is_hallucination,
        "explanation": explanation,
        "original_answer": initial_answer
    }
    
    return final_answer, retrieved_docs, hallucination_info