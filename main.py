"""
LangChain を使った RAG (Retrieval-Augmented Generation) サンプル
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain のインポート
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# 環境変数の読み込み
load_dotenv()

class RAGSystem:
    """RAG システムのメインクラス"""
    
    def __init__(self, openai_api_key: str = None):
        """
        RAG システムの初期化
        
        Args:
            openai_api_key: OpenAI API キー
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY が設定されていません")
            
        # OpenAI モデルの初期化
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )
        
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=self.openai_api_key
        )
        
        # テキスト分割器の設定
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Chroma ベクトルストアの初期化
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self, file_path: str) -> List[Document]:
        """
        テキストファイルからドキュメントを読み込み
        
        Args:
            file_path: ファイルパス
            
        Returns:
            Document のリスト
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # ドキュメントの作成
            doc = Document(
                page_content=content,
                metadata={"source": file_path}
            )
            
            return [doc]
            
        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        ドキュメントを小さなチャンクに分割
        
        Args:
            documents: 分割するドキュメントのリスト
            
        Returns:
            分割されたドキュメントのリスト
        """
        return self.text_splitter.split_documents(documents)
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """
        ベクトルストアを作成してドキュメントを保存
        
        Args:
            documents: インデックス化するドキュメントのリスト
        """
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            print(f"ベクトルストアを作成しました。ドキュメント数: {len(documents)}")
            
        except Exception as e:
            print(f"ベクトルストア作成エラー: {e}")
            raise
    
    def create_qa_chain(self) -> None:
        """
        QA チェーンを作成
        """
        if not self.vectorstore:
            raise ValueError("ベクトルストアが作成されていません")
            
        # カスタムプロンプトテンプレートの作成
        prompt_template = """以下のコンテキストを使用して質問に答えてください。
        コンテキストにない情報については「コンテキストに情報がありません」と回答してください。
        
        コンテキスト: {context}
        
        質問: {question}
        
        回答:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RetrievalQA チェーンの作成
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        質問に回答
        
        Args:
            question: 質問文
            
        Returns:
            回答とソース文書を含む辞書
        """
        if not self.qa_chain:
            raise ValueError("QA チェーンが作成されていません")
            
        try:
            result = self.qa_chain({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
            
        except Exception as e:
            print(f"質問処理エラー: {e}")
            return {"answer": "エラーが発生しました", "source_documents": []}

def main():
    """メイン関数"""
    print("🚀 RAG システムを開始しています...")
    
    # API キーの確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY が設定されていません")
        print("💡 以下の方法で API キーを設定してください：")
        print("   1. .env ファイルに OPENAI_API_KEY=your_key_here を追加")
        print("   2. 環境変数として export OPENAI_API_KEY=your_key_here")
        return
    
    try:
        # RAG システムの初期化
        rag_system = RAGSystem()
        
        # ドキュメントの読み込み
        print("📄 ドキュメントを読み込んでいます...")
        documents = rag_system.load_documents("data/example.txt")
        
        if not documents:
            print("❌ ドキュメントの読み込みに失敗しました")
            return
            
        # ドキュメントの分割
        print("✂️ ドキュメントを分割しています...")
        split_docs = rag_system.split_documents(documents)
        print(f"📝 {len(split_docs)} 個のチャンクに分割しました")
        
        # ベクトルストアの作成
        print("🔍 ベクトルストアを作成しています...")
        rag_system.create_vectorstore(split_docs)
        
        # QA チェーンの作成
        print("🔗 QA チェーンを作成しています...")
        rag_system.create_qa_chain()
        
        print("\n✅ RAG システムの準備が完了しました！")
        print("=" * 50)
        
        # サンプル質問の実行
        sample_questions = [
            "LangChain とは何ですか？",
            "RAG の利点を教えてください",
            "Chroma の特徴は何ですか？",
            "OpenAI Embeddings にはどのようなモデルがありますか？"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n🤔 質問 {i}: {question}")
            print("-" * 30)
            
            result = rag_system.ask_question(question)
            print(f"🤖 回答: {result['answer']}")
            
            if result['source_documents']:
                print(f"📚 参照元: {len(result['source_documents'])} 個の文書")
                for j, doc in enumerate(result['source_documents'][:2]):  # 最初の2つのみ表示
                    print(f"   - 文書 {j+1}: {doc.page_content[:100]}...")
            
            print()
        
        print("=" * 50)
        print("🎉 動作確認が完了しました！")
        
        # インタラクティブモード
        print("\n💬 インタラクティブモードに入ります（'quit' で終了）")
        while True:
            user_question = input("\n質問を入力してください: ")
            if user_question.lower() in ['quit', 'exit', '終了']:
                break
                
            result = rag_system.ask_question(user_question)
            print(f"🤖 回答: {result['answer']}")
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return

if __name__ == "__main__":
    main()