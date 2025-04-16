# test.py for testing purpose no use 
from retrieval import AccessControlledRetriever, set_agent_clearance

retriever = AccessControlledRetriever()
set_agent_clearance(8)

query = "are there an warehouses"
results = retriever.retrieve(query,use_llm=True)

print("------ RESULTS ------")
print(results)

