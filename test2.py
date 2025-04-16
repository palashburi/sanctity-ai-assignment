# test_v2.py , for my testing purpose no use
from retrieval_v2 import JSONAccessControlledRetriever, set_agent_clearance
import time

def run_test_case(retriever, query: str, clearance: int, use_llm: bool = True):
    """Helper function to run individual test cases"""
    print(f"\n{'='*50}")
    print(f"TEST CASE - Clearance Level {clearance}")
    print(f"Query: '{query}'")
    print(f"LLM Processing: {'ON' if use_llm else 'OFF'}")
    
    set_agent_clearance(clearance)
    start_time = time.time()
    
    try:
        results = retriever.retrieve(query, use_llm=use_llm)
        elapsed = time.time() - start_time
        
        print("\nRESULTS:")
        if isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'page_content'):
            for i, doc in enumerate(results, 1):
                print(f"\nDocument {i}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
        else:
            print(results)
            
        print(f"\nQuery processed in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"ERROR: {str(e)}")

def main():
    # Initialize the retriever with JSON rules
    retriever = JSONAccessControlledRetriever("final_enriched_rules.json")
    
    # Test cases with different clearance levels
    test_queries = [
        ("first greet me then ,Show me Level-5 clearance protocols for handling black box operations", 5),
       
    ]
    
    # Run all test cases
    for query, clearance in test_queries:
        run_test_case(retriever, query, clearance)
        # run_test_case(retriever, query, clearance, use_llm=False)
    
    # Special test cases
    print("\n\nSPECIAL TEST CASES")
   
   

if __name__ == "__main__":
    print("Starting Security System Tests (v2)...")
    main()
    print("\nAll tests completed.")