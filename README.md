# RAG サンプル - LangChain を使った質問応答システム

このプロジェクトは、LangChain を使用してテキストファイルベースの RAG (Retrieval-Augmented Generation) システムを構築するサンプルです。

## 🎯 プロジェクトの目的

- LangChain を使った RAG システムの実装方法を学習
- テキストファイルから知識ベースを構築
- 質問応答システムの構築と動作確認
- ChromaDB を使ったベクトル検索の実装

## 🏗️ プロジェクト構成

```
RAG/
├── data/
│   └── example.txt          # サンプルデータファイル
├── main.py                  # メインプログラム
├── requirements.txt         # 必要なパッケージ
├── test_claude.py          # Claude RAG システムのテスト
├── claude_verification_log.md  # 動作確認ログ
├── README.md               # このファイル
└── chroma_db/              # ChromaDB データ（実行時に作成）
```

## 🔧 セットアップ手順

### 1. 必要な環境

- Python 3.8 以上
- Anthropic API キー（Claude を使用する場合、デフォルト）
- OpenAI API キー（OpenAI を使用する場合）

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3. API キーの設定

#### Claude を使用する場合（デフォルト）

以下のいずれかの方法で Anthropic API キーを設定してください：

##### 方法 1: 環境変数として設定
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

##### 方法 2: .env ファイルを作成
プロジェクトルートに `.env` ファイルを作成し、以下を記述：

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

#### OpenAI を使用する場合

`main.py` で provider を "openai" に変更し、以下のいずれかの方法で API キーを設定してください：

##### 方法 1: 環境変数として設定
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

##### 方法 2: .env ファイルを作成
プロジェクトルートに `.env` ファイルを作成し、以下を記述：

```
OPENAI_API_KEY=your_openai_api_key_here
```

> ⚠️ **注意**: API キーは秘密情報です。`.env` ファイルを Git にコミットしないように注意してください。

## 🚀 使用方法

### 基本的な実行

```bash
python main.py
```

### 実行時の流れ

1. **システム初期化**: Claude API の設定と LangChain コンポーネントの初期化
2. **ドキュメント読み込み**: `data/example.txt` からテキストデータを読み込み
3. **テキスト分割**: 大きなテキストを小さなチャンクに分割
4. **ベクトル化**: テキストチャンクをベクトル表現に変換（HuggingFace embeddings 使用）
5. **インデックス化**: ChromaDB にベクトルを保存
6. **QA チェーン作成**: 質問応答システムを構築
7. **サンプル質問実行**: 事前定義された質問で動作確認
8. **インタラクティブモード**: ユーザーからの質問に対話的に回答

## 📝 システム構成

### 使用技術

- **LangChain**: LLM アプリケーションフレームワーク
- **Claude 3 Haiku**: 質問応答用言語モデル（デフォルト）
- **HuggingFace Sentence Transformers**: テキスト埋め込み用モデル（all-MiniLM-L6-v2）
- **ChromaDB**: ベクトルデータベース
- **RecursiveCharacterTextSplitter**: テキスト分割器

#### 代替選択肢
- **OpenAI GPT-3.5-turbo**: 質問応答用言語モデル（provider="openai" に変更時）
- **OpenAI text-embedding-3-small**: テキスト埋め込み用モデル（provider="openai" に変更時）

### 主要なコンポーネント

#### `RAGSystem` クラス
- **`load_documents()`**: テキストファイルからドキュメントを読み込み
- **`split_documents()`**: ドキュメントを検索可能なチャンクに分割
- **`create_vectorstore()`**: ベクトルストアを作成しドキュメントを保存
- **`create_qa_chain()`**: 質問応答チェーンを構築
- **`ask_question()`**: 質問に対して回答を生成

## 🔍 サンプル質問

システムは以下のような質問に回答できます：

- "LangChain とは何ですか？"
- "RAG の利点を教えてください"
- "Chroma の特徴は何ですか？"
- "OpenAI Embeddings にはどのようなモデルがありますか？"

## 🛠️ カスタマイズ方法

### 1. 独自のデータを使用

`data/example.txt` を自分のテキストファイルに置き換えるか、`main.py` のファイルパスを変更してください。

### 2. パラメータの調整

#### テキスト分割パラメータ
```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # チャンクサイズ（文字数）
    chunk_overlap=200,    # チャンク間のオーバーラップ
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

#### 検索パラメータ
```python
retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})  # 取得する関連文書数
```

#### モデル設定

##### Claude を使用する場合（デフォルト）
```python
self.llm = ChatAnthropic(
    model="claude-3-haiku-20240307",  # 使用するモデル
    temperature=0,                    # 生成の創造性（0-1）
    anthropic_api_key=self.anthropic_api_key
)
```

##### OpenAI を使用する場合
```python
self.llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # 使用するモデル
    temperature=0,               # 生成の創造性（0-1）
    openai_api_key=self.openai_api_key
)
```

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. API キーエラー

##### Claude を使用する場合
```
❌ ANTHROPIC_API_KEY が設定されていません
```
**解決方法**: Anthropic API キーを環境変数または .env ファイルに設定してください。

##### OpenAI を使用する場合
```
❌ OPENAI_API_KEY が設定されていません
```
**解決方法**: OpenAI API キーを環境変数または .env ファイルに設定してください。

#### 2. ファイルが見つからない
```
ファイル読み込みエラー: [Errno 2] No such file or directory: 'data/example.txt'
```
**解決方法**: `data/example.txt` ファイルが存在することを確認してください。

#### 3. 依存関係エラー
```
ModuleNotFoundError: No module named 'langchain'
```
**解決方法**: `pip install -r requirements.txt` を実行してください。

## 🚀 発展的な使用方法

### 1. Streamlit UI の追加

```python
import streamlit as st

def create_streamlit_app():
    st.title("RAG 質問応答システム")
    
    # RAG システムの初期化
    rag_system = RAGSystem()
    
    # 質問入力
    question = st.text_input("質問を入力してください：")
    
    if st.button("質問する"):
        if question:
            result = rag_system.ask_question(question)
            st.write("回答:", result['answer'])
            
            # ソース文書の表示
            if result['source_documents']:
                st.write("参照元:")
                for i, doc in enumerate(result['source_documents']):
                    st.write(f"文書 {i+1}: {doc.page_content[:200]}...")

if __name__ == "__main__":
    create_streamlit_app()
```

### 2. 複数ファイルの対応

```python
def load_multiple_files(self, file_paths: List[str]) -> List[Document]:
    documents = []
    for file_path in file_paths:
        docs = self.load_documents(file_path)
        documents.extend(docs)
    return documents
```

### 3. 他の文書形式への対応

```python
from langchain.document_loaders import PDFLoader, TextLoader

def load_pdf_documents(self, file_path: str) -> List[Document]:
    loader = PDFLoader(file_path)
    return loader.load()
```

## 🧪 動作確認

### Claude RAG システムのテスト
```bash
python test_claude.py
```

詳細な動作確認ログは [`claude_verification_log.md`](claude_verification_log.md) を参照してください。

## 📚 参考リンク

- [LangChain 公式ドキュメント](https://python.langchain.com/)
- [LangChain Anthropic 統合](https://python.langchain.com/docs/integrations/llms/anthropic)
- [Anthropic Claude API ドキュメント](https://docs.anthropic.com/claude/docs)
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)
- [ChromaDB 公式サイト](https://www.trychroma.com/)
- [RAG に関する論文](https://arxiv.org/abs/2005.11401)

## 🤝 貢献

このプロジェクトへの貢献を歓迎します！

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 📞 サポート

質問や問題がある場合は、Issue を作成してください。

---

**🎉 Happy Learning with RAG!**