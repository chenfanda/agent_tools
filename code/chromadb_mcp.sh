export CHROMA_CLIENT_TYPE="http"
export CHROMA_COLLECTION_NAME="my_ollama_documents"
export CHROMA_EMBEDDING_FUNCTION="ollama"
export CHROMA_HOST='localhost'
export CHROMA_PORT=8080
export OLLAMA_MODEL='mxbai-embed-large'
export OLLAMA_HOST='http://localhost:11434'
mcp dev chromadb_mcp.py
