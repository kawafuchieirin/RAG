#!/usr/bin/env python3
"""
Claude/Anthropic RAG システムのテストスクリプト
"""

import os
import sys
from pathlib import Path

# スクリプトディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from main import RAGSystem

def test_claude_integration():
    """Claude 統合の基本テスト"""
    
    print("🧪 Claude RAG システムのテストを開始します...")
    
    # API キーの確認
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY が設定されていません")
        print("💡 テスト実行前に API キーを設定してください")
        return False
    
    try:
        # RAG システムの初期化
        print("🔧 RAG システムを初期化しています...")
        rag_system = RAGSystem(provider="claude")
        
        # シンプルなテスト用ドキュメント作成
        test_content = """
        Claude は Anthropic によって開発された大規模言語モデルです。
        Claude は以下の特徴を持っています：
        - 高度な推論能力
        - 安全性への配慮
        - 長いコンテキストの処理能力
        - 多言語対応
        """
        
        # テスト用ファイルの作成
        test_file = "test_data.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        # ドキュメントの読み込み
        print("📄 テストドキュメントを読み込んでいます...")
        documents = rag_system.load_documents(test_file)
        
        if not documents:
            print("❌ ドキュメントの読み込みに失敗しました")
            return False
        
        # ドキュメントの分割
        print("✂️ ドキュメントを分割しています...")
        split_docs = rag_system.split_documents(documents)
        
        # ベクトルストアの作成
        print("🔍 ベクトルストアを作成しています...")
        rag_system.create_vectorstore(split_docs)
        
        # QA チェーンの作成
        print("🔗 QA チェーンを作成しています...")
        rag_system.create_qa_chain()
        
        # テスト質問の実行
        test_question = "Claude の特徴を教えてください"
        print(f"🤔 テスト質問: {test_question}")
        
        result = rag_system.ask_question(test_question)
        
        print(f"🤖 回答: {result['answer']}")
        
        if result['source_documents']:
            print(f"📚 参照された文書数: {len(result['source_documents'])}")
        
        # クリーンアップ
        if os.path.exists(test_file):
            os.remove(test_file)
            
        print("\n✅ Claude RAG システムのテストが完了しました！")
        return True
        
    except Exception as e:
        print(f"❌ テスト中にエラーが発生しました: {e}")
        # クリーンアップ
        if os.path.exists("test_data.txt"):
            os.remove("test_data.txt")
        return False

def main():
    """メイン関数"""
    success = test_claude_integration()
    
    if success:
        print("\n🎉 すべてのテストが正常に完了しました！")
        print("💡 メインスクリプトを実行してください: python main.py")
    else:
        print("\n❌ テストに失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()