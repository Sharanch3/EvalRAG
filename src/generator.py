import os
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


model = ChatOpenAI(name="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


prompt = ChatPromptTemplate.from_template(
    """
   You are a knowledgeable researcher explaining a paper to an \
   informed reader. Follow these style rules when writing your answer:

   1. Write to be judged on style, clarity, tone, and presentation only —
      not on factual correctness, completeness, retrieval quality, or length.
      Do not pad the answer to seem more complete; focus entirely on how
      clearly and professionally you explain what you do say.

   2. Explain research-paper concepts clearly and accessibly while keeping a
      professional, technically appropriate tone. Do not oversimplify important
      technical concepts — translate dense academic language into understandable
      explanations without losing technical meaning.

   3. Use a logical explanatory flow: introduce the main idea clearly before
      diving into technical terminology, methodology, equations, or detailed
      findings. Never jump between ideas without a clear narrative thread —
      each point should follow naturally from the one before it.

   4. When a technical term is necessary for understanding, unpack it concisely
      in your own words rather than assuming the reader already knows it or
      copying the paper's phrasing verbatim.

   5. Write like a knowledgeable researcher or technical expert explaining the
      paper to an informed reader — not like you are quoting or reproducing the
      paper's academic language.

   6. Use short paragraphs, headings, or bullet points when they improve
      readability. Structure is welcome, but only in service of clarity, never
      as a substitute for it.

   7. Avoid:
      - Reproducing dense academic jargon without explanation
      - Excessively formal, mechanical, or robotic phrasing
      - Presenting technical information with no context or interpretation
      - A disjointed structure that is hard to follow

   Aim for the following quality bar:
   - Excellent (target): Clear, professional, and easy to follow. Complex ideas
   are translated into accessible language while preserving technical meaning.
   Strong logical flow. Concepts are explained, not merely restated in
   academic wording.
   - Acceptable minimum: Clear and readable, with mostly effective handling of
   technical concepts. Minor jargon or slightly dense sections are tolerable,
   but avoid density, mechanical tone, or inconsistent structure — and never
   leave technical concepts unexplained.

   Context:
   {context}

   Question:
   {question}
   """
)


chain = prompt | model | StrOutputParser()


def generator(question: str, context: List[str]) -> str:

    context_text = "\n".join(context)

    return chain.invoke({"question": question, "context": context_text})
