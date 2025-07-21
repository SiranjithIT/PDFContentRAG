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
      
      print(f"Loaded {len(self.docs)} pages from PDF")
      
      if self.docs:
    
        for i, doc in enumerate(self.docs):
          content = doc.page_content
          lines = content.split('\n')
          cleaned_lines = []
          for line in lines:
            line = line.strip()
            if len(line) > 10:  # Filter out very short lines
              cleaned_lines.append(line)
          
          doc.page_content = '\n'.join(cleaned_lines)
          print(f"Page {i+1} cleaned content length: {len(doc.page_content)}")
      
      # Add metadata to all documents
      for doc in self.docs:
        doc.metadata['doc_id'] = self._generate_doc_id(path)
        doc.metadata['source_file'] = path
      
      # Split all documents
      text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
      )
      
      self.text_splits = text_splitter.split_documents(self.docs)
      print(f"Created {len(self.text_splits)} text splits from all pages")
      
      # Debug: Show first few splits
      if self.text_splits:
        print("First split preview:")
        print(self.text_splits[0].page_content[:300] + "...")
        
    except Exception as e:
      print(f"Error loading or splitting the text: {e}")
      import traceback
      traceback.print_exc()
      self.text_splits = []
   
  def _generate_doc_id(self, path):
    return hashlib.md5(path.encode()).hexdigest()
   
  def get_splits(self):
    return self.text_splits

class Retriever(TextSplit):
  def __init__(self, embeddings=embeddings, path=path):
    super().__init__(path)
    self.text_split = super().get_splits()
    self.path = path
    self.vector_store = Chroma(
      collection_name="pdf_content",  # Changed collection name
      embedding_function=embeddings,
      persist_directory="./chroma_db",
    )
    
    if self.text_split and not self._document_exist(path):
      print(f"Adding {len(self.text_split)} documents to vector store")
      self.vector_store.add_documents(documents=self.text_split)
    elif self._document_exist(path):
      print("Documents already exist in vector store")
    else:
      print("No text splits to add to vector store")
  
  def _document_exist(self, path):
    try:
      doc_id = self._generate_doc_id(path)
      
      # Better approach: use get() method
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
      print(f"Querying for: '{query_text}'")
      results = self.vector_store.similarity_search(query_text, k=k)
      print(f"Found {len(results)} results")
      return results
    except Exception as e:
      print(f"Error querying vector store: {e}")
      return []
  
  def delete_collection(self):
    try:
      self.vector_store.delete_collection()
      print("Collection deleted successfully")
    except Exception as e:
      print(f"Error deleting collection: {e}")

  def debug_collection(self):
    """Debug method to see what's in the collection"""
    try:
      results = self.vector_store.get()
      print(f"Total documents in collection: {len(results['ids'])}")
      
      if results['documents']:
        print("Sample document content:")
        for i, doc in enumerate(results['documents'][:2]):
          print(f"Doc {i+1}: {doc[:200]}...")
            
    except Exception as e:
      print(f"Error debugging collection: {e}")

# Test the PDF loader
if __name__ == "__main__":
  print("Testing PDF loader...")
  
  # Check if file exists
  if not os.path.exists(path):
    print(f"ERROR: File not found at {path}")
    print("Current working directory:", os.getcwd())
    print("Please check the file path!")
  else:
    print(f"File found at: {path}")
    
    # Test loading
    retriever = Retriever()
    
    # Debug the collection
    retriever.debug_collection()
    
    # Test a query
    if retriever.text_split:
      results = retriever.query("Judgement", k=3)
      for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(result.page_content[:300] + "...")
        print(f"Metadata: {result.metadata}")
    else:
      print("No text splits available for querying")