# dev_retriever.py
# DEVELOPMENT PHASE 2: Agent development utilities
# Import this in your agent notebooks for easy retriever access

import pandas as pd
import json
import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes import config as wvcc
from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage
from typing import List, Dict
import os

# Configuration
COLLECTION_NAME = "GAIADocuments"
PERSISTENCE_PATH = "./weaviate_data"

class DevelopmentGAIARetriever:
    """
    Development-focused GAIA retriever
    Optimized for notebook experimentation and agent development
    """
    
    def __init__(self, csv_file: str = 'gaia_embeddings.csv'):
        self.csv_file = csv_file
        self.client = None
        self.vector_store = None
        self.retriever = None
        self.embeddings = None
        
        # Check if vector store exists
        if not os.path.exists(csv_file):
            print(f"❌ Vector store not found: {csv_file}")
            print("💡 Run 'python build_vectorstore.py' first")
            return
        
        print(f"📂 Found vector store: {csv_file}")
    
    def setup(self, quick_mode: bool = True):
        """
        Setup retriever for development
        quick_mode: Skip some setup for faster notebook iterations
        """
        if not os.path.exists(self.csv_file):
            print("❌ Vector store not built. Run build_vectorstore.py first.")
            return False
        
        print("🔄 Setting up development retriever...")
        
        # Setup embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 8}
        )
        
        # Create embedded Weaviate client
        self.client = self._create_embedded_client()
        if not self.client:
            return False
        
        # Load CSV data if not already loaded
        if not self._is_data_loaded():
            if not self._load_csv_data():
                return False
        else:
            print("✅ Data already loaded, skipping CSV import")
        
        # Create LangChain retriever
        self.vector_store = WeaviateVectorStore(
            client=self.client,
            index_name=COLLECTION_NAME,
            text_key="content",
            embedding=self.embeddings,
            attributes=["source"]
        )
        
        self.retriever = self.vector_store.as_retriever()
        
        print("✅ Development retriever ready!")
        return True
    
    def _create_embedded_client(self):
        """Create embedded Weaviate client for development"""
        try:
            client = weaviate.connect_to_embedded(
                version="1.26.5",
                persistence_data_path=PERSISTENCE_PATH,
                binary_path="./weaviate_binary",
                port=8080,
                grpc_port=50051,
                environment_variables={
                    "QUERY_DEFAULTS_LIMIT": "25",
                    "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true"
                }
            )
            
            if client.is_ready():
                print("✅ Embedded Weaviate started")
                return client
            else:
                print("❌ Embedded Weaviate failed to start")
                return None
                
        except Exception as e:
            print(f"❌ Embedded Weaviate error: {e}")
            return None
    
    def _is_data_loaded(self) -> bool:
        """Check if data is already loaded in Weaviate"""
        try:
            if not self.client.collections.exists(COLLECTION_NAME):
                return False
            
            collection = self.client.collections.get(COLLECTION_NAME)
            count = collection.aggregate.over_all(total_count=True).total_count
            return count > 0
            
        except:
            return False
    
    def _load_csv_data(self) -> bool:
        """Load CSV data into Weaviate"""
        try:
            print("📂 Loading CSV data...")
            
            # Create collection
            if self.client.collections.exists(COLLECTION_NAME):
                self.client.collections.delete(COLLECTION_NAME)
            
            collection = self.client.collections.create(
                name=COLLECTION_NAME,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                ],
                vectorizer_config=wvcc.Configure.Vectorizer.none(),
                vector_index_config=wvcc.Configure.VectorIndex.hnsw(),
            )
            
            # Load CSV
            df = pd.read_csv(self.csv_file)
            print(f"  ├── Loading {len(df)} documents...")
            
            with collection.batch.dynamic() as batch:
                for _, row in df.iterrows():
                    metadata = json.loads(row['metadata'])
                    embedding_vector = json.loads(row['embedding'])
                    
                    batch.add_object(
                        properties={
                            "content": row['content'],
                            "source": metadata['source']
                        },
                        vector=embedding_vector
                    )
            
            total_objects = collection.aggregate.over_all(total_count=True).total_count
            print(f"✅ Loaded {total_objects} documents")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if retriever is ready for use"""
        return self.vector_store is not None
    
    def search(self, question: str, k: int = 1):
        """Search for similar GAIA examples"""
        if not self.is_ready():
            print("❌ Retriever not ready. Call setup() first.")
            return []
        
        return self.vector_store.similarity_search(question, k=k)
    
    def retriever_node(self, state_messages: list):
        """
        #1 student's retriever node implementation
        Use this in your agent development
        """
        if not self.is_ready():
            return {"messages": state_messages}
        
        try:
            question = state_messages[0].content if state_messages else ""
            
            # Exact same as #1 student
            similar_question = self.vector_store.similarity_search(question)
            
            example_msg_content = f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}"
            
            example_msg = HumanMessage(content=example_msg_content)
            
            return {"messages": state_messages + [example_msg]}
            
        except Exception as e:
            print(f"❌ Retriever error: {e}")
            return {"messages": state_messages}
    
    def test_retrieval(self, test_questions: List[str] = None):
        """Test retrieval with sample questions"""
        if not self.is_ready():
            print("❌ Retriever not ready")
            return
        
        if test_questions is None:
            test_questions = [
                "Calculate compound interest",
                "Find information about a scientific paper", 
                "Analyze an Excel spreadsheet"
            ]
        
        print("🧪 Testing retrieval...")
        print("=" * 40)
        
        for question in test_questions:
            print(f"\n🔍 Query: '{question}'")
            results = self.search(question, k=1)
            if results:
                content = results[0].page_content
                print(f"✅ Found: {content[:100]}...")
            else:
                print("❌ No results")
    
    def close(self):
        """Clean shutdown"""
        if self.client:
            self.client.close()
            print("✅ Retriever closed")

# Convenience function for notebook use
def load_gaia_retriever(csv_file: str = 'gaia_embeddings.csv') -> DevelopmentGAIARetriever:
    """
    Quick loader for notebooks
    Usage: retriever = load_gaia_retriever()
    """
    retriever = DevelopmentGAIARetriever(csv_file)
    
    if retriever.setup():
        print("🎯 Retriever ready for agent development!")
        return retriever
    else:
        print("❌ Failed to setup retriever")
        return None

# Example notebook usage
"""
# In your agent development notebooks:

from dev_retriever import load_gaia_retriever

# Quick setup
retriever = load_gaia_retriever()

# Test it works
retriever.test_retrieval()

# Use in your agent (exactly like #1 student)
def my_agent_retriever_node(state_messages):
    return retriever.retriever_node(state_messages)

# Experiment with different approaches
similar_examples = retriever.search("My test question", k=3)
print(similar_examples[0].page_content)
"""