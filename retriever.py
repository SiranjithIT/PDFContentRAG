from dotenv import load_dotenv
import hashlib
import os

load_dotenv()

if not os.getenv("USER_AGENT"):
  os.environ["USER_AGENT"] = "myagent"

from models import *
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import bs4

path = "Judgement.pdf"
class TextSplit:
  def __init__(self, path):
    try:
      loader = PyPDFLoader(file_path=path)
      self.docs = loader.load()
      
      if self.docs:
        for i, doc in enumerate(self.docs):
          content = doc.page_content
          lines = content.split('\n')
          cleaned_lines = []
          for line in lines:
            line = line.strip()
            if len(line) > 10:
              cleaned_lines.append(line)
          
          doc.page_content = '\n'.join(cleaned_lines)
          print(f"Page {i+1} cleaned content length: {len(doc.page_content)}")
      
      for doc in self.docs:
        doc.metadata['doc_id'] = self._generate_doc_id(path)
        doc.metadata['source'] = path
      
      text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap=200,
        add_start_index=True,
      )
      
      self.text_splits = text_splitter.split_documents(self.docs)
    except Exception as e:
      print(f"Error loading or splitting the text")
      self.text_splits = []
   
  def _generate_doc_id(self, path):
    return hashlib.md5(path.encode()).hexdigest()
   
  def get_splits(self):
    return self.text_splits

class Retriever(TextSplit):
  def __init__(self,embeddings=embeddings , path=path):
    super().__init__(path)
    self.text_split = super().get_splits()
    self.vector_store = Chroma(
      collection_name="web_content",
      embedding_function=embeddings,
      persist_directory="./chroma_db",
    )
    
    if self.text_split and not self._document_exist(path):
      self.vector_store.add_documents(documents=self.text_split)
  
  def _document_exist(self, path):
    try:
      doc_id = self._generate_doc_id(path)
      results = self.vector_store.get(where={"doc_id": doc_id})
      return len(results['ids']) > 0 if results else False
    except Exception as e:
      print(f"Unable to check the datastore: {e}")
      return False
      
  def get_retriever(self, search_type="similarity", k=4):
    return self.vector_store.as_retriever(
      search_type=search_type,
      search_kwargs={"k": k}
    )
    
  def query(self, query_text, k=4):
    try:
      results = self.vector_store.similarity_search(query_text, k=k)
      return results
    except Exception as e:
      print(f"Error querying vector store: {e}")
      return []
  
  def delete_collection(self):
    try:
      self.vector_store.delete_collection()
      print(f"Collection '{self.collection_name}' deleted successfully")
    except Exception as e:
      print(f"Error deleting collection: {e}")