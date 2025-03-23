from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def retrieveDocument(state):
    print("---RETRIEVE---")
    question = state["question"]
    if not state["counter"]:
        counter = 0
    else:
        counter = state["counter"]
    print("Question:",question)
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question,"counter": counter + 1}