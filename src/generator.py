import os
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


model = ChatOpenAI(name="gpt-4o-mini", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"))


prompt = ChatPromptTemplate.from_template(
    """
    You are a research assistant who explains research papers clearly,
    accurately, and in a beginner-friendly way.

    Your job is to answer the user's question using ONLY the information
    provided in the research paper context.

    Rules:

    1. Use ONLY the provided context.
       Do not use outside knowledge, even if you are confident that it is correct.

    2. Do not hallucinate.
       If the context does not contain enough information to answer the question,
       say:
       "The provided research paper context does not contain enough information
       to answer this question."

    3. Explain concepts in simple language first.
       If the paper contains technical terms, explain what they mean before
       going into technical details.

    4. Explain the "why" behind the research.
       When the context provides enough information, explain:
       - What problem the researchers are trying to solve
       - Why the problem is important
       - What approach or method they proposed
       - How the method works
       - What experiments or evaluations were performed
       - What results were obtained
       - What conclusions the researchers reached

    5. When explaining a technical method, follow this order:
       Idea → Intuition → How it works → Technical details → Result

    6. Use examples or analogies only when they are directly supported
       by the context. Do not introduce facts that are not present in the paper.

    7. Distinguish clearly between:
       - What the paper explicitly states
       - What can be directly concluded from the provided context

    8. If the question asks for a comparison, compare only the information
       available in the context.

    9. If the user asks about a formula, algorithm, architecture, experiment,
       or result, explain it step by step using the information available
       in the context.

    10. Keep the explanation conversational and easy to understand.
        Avoid unnecessarily complicated academic language.

    Research Paper Context:
    {context}

    Question:
    {question}

    """
)


chain = prompt | model | StrOutputParser()


def generator(question: str, context: List[str]) -> str:

    context_text = "\n".join(context)

    return chain.invoke({"question": question, "context": context_text})
