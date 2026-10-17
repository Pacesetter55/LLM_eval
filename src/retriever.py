from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EMBEDDING_MODEL = "qwen3-embedding:8b"
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 250
# Qwen3-Embedding is instruction-aware: queries get a task prefix, documents are embedded as-is.
QUERY_INSTRUCTION = "Given a question about a health insurance policy, retrieve the policy passages that answer it"

# Vectors from different embedding models aren't comparable (or even the same size),
# so each model gets its own store; switching models never reads a stale index.
PERSIST_DIR = Path(__file__).resolve().parent.parent / "vectorstore" / EMBEDDING_MODEL.replace(":", "-")


class QwenEmbeddings(OllamaEmbeddings):
    """OllamaEmbeddings that adds Qwen3-Embedding's instruction prefix to queries only."""

    def _with_instruction(self, text: str) -> str:
        return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(self._with_instruction(text))

    async def aembed_query(self, text: str) -> list[float]:
        return await super().aembed_query(self._with_instruction(text))


embeddings = QwenEmbeddings(model=EMBEDDING_MODEL)


def get_retriever(k: int = 4):
    db_already_built = PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())

    if db_already_built:
        vectorstore = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)
    else:
        loader = DirectoryLoader(str(DATA_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIR))

    return vectorstore.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    retriever = get_retriever()
    results = retriever.invoke("What does Ayush day care means?")
    for doc in results:
        print(doc.page_content[:200])
        print("---")
