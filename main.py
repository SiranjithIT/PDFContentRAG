from retriever import Retriever
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, List, Any
from models import llm
from langchain_core.prompts import ChatPromptTemplate

class AgentState(TypedDict):
  question: str
  context: List[Any]
  answer: str
  
def retriever(state: AgentState)->AgentState:
  retrieve = Retriever()
  context = retrieve.query(state['question'], k = 7)
  state['context'].extend(context)
  return state

def generator(state: AgentState)->AgentState:
  prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an helpful Agent assisting the user with their query. Use the context provided to answer the query"),
    ("human","""
     Context: {context}
     Question: {question}
     
     Please provide the comprehensive answer for the question based on the context.
     """)
  ])
  
  chain = prompt | llm
  context_txt = " ".join([doc.page_content for doc in state['context']])
  
  response = chain.invoke({
    "context": context_txt,
    "question": state["question"]
  })
  
  if hasattr(response, "content"):
    state["answer"] = response.content
  else:
    state["answer"] = str(response)
  return state

graph = StateGraph(AgentState)

graph.add_node("retrieve", retriever)
graph.add_node("generate", generator)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve","generate")
graph.add_edge("generate", END)

RAGAgent = graph.compile()

def main():
  while True:
    query_input = input("Enter your query('exit' to quit): ")
    if 'exit' in query_input.lower():
      break
    try:
      result = RAGAgent.invoke(AgentState(
        question=query_input,
        context=[],
        answer=""
      ))
      
      print(f"\nAnswer: {result['answer']}")
      print(f"\nUsed {len(result['context'])} context documents")
    except Exception as err:
      print("Error: ", err)

if __name__ == "__main__":
  main()


    
