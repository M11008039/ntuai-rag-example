# Telegram RAG Chatbot

這是一個給技術新手上手的 Telegram RAG chatbot 範例。你可以把 PDF、DOCX、TXT、Markdown 文件傳給 Telegram bot，bot 會把文件切成 chunks、用 Gemini embedding 建立向量索引，之後就能直接在 Telegram 裡詢問文件內容。

## 功能

- Telegram 對話式介面
- 支援上傳 `PDF`、`DOCX`、`TXT`、`MD`
- 使用 Gemini `gemini-embedding-001` 建立 embedding
- 使用 ChromaDB 本地持久化向量資料庫
- 使用 Gemini `gemini-2.5-flash` 根據檢索內容回答
- 回答會附上來源檔名、頁碼或 chunk
- 支援 Docker Compose 一鍵啟動
- 內建 pytest 測試

## 架構

```text
Telegram 使用者
      ↓
python-telegram-bot
      ↓
文件下載到 data/uploads
      ↓
pypdf / python-docx / text reader
      ↓
LangChain RecursiveCharacterTextSplitter
      ↓
Gemini embedding
      ↓
ChromaDB data/chroma
      ↓
使用者提問
      ↓
ChromaDB similarity search
      ↓
Gemini 2.5 Flash 回答
      ↓
Telegram 顯示答案與來源
```

## 專案結構

```text
.
├── src/telegram_rag_chatbot/
│   ├── bot.py          # Telegram bot 入口與指令
│   ├── chunking.py     # 文件切 chunk
│   ├── config.py       # .env 設定
│   ├── loaders.py      # PDF / DOCX / TXT / MD 讀取
│   ├── prompts.py      # RAG prompt 與來源格式
│   └── rag.py          # Gemini + ChromaDB RAG 流程
├── tests/              # pytest 測試
├── data/uploads/       # Telegram 上傳文件
├── data/chroma/        # ChromaDB 本地資料
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## 事前準備

你需要兩個 key：

1. Gemini API key  
   到 Google AI Studio 建立 API key。

2. Telegram bot token  
   在 Telegram 找 `@BotFather`，建立 bot 後取得 token。

## Docker Compose 啟動

先準備 `.env`：

```bash
cp .env.example .env
```

填入 `GEMINI_API_KEY` 和 `TELEGRAM_BOT_TOKEN` 後啟動：

```bash
docker compose up -d --build
```

查看 logs：

```bash
docker compose logs -f bot
```

停止：

```bash
docker compose down
```

接著到 Telegram 打開你的 bot：

1. 輸入 `/start`
2. 第一次啟動時，bot 會自動索引 `data/uploads` 內的文件
3. 直接詢問文件內容
4. 也可以上傳 PDF、DOCX、TXT 或 MD 追加到知識庫

## 本機啟動 (若無法使用 Docker)

複製環境變數範例：

```bash
cp .env.example .env
```

編輯 `.env`：

```env
GEMINI_API_KEY=你的_Gemini_API_Key
TELEGRAM_BOT_TOKEN=你的_Telegram_Bot_Token
```

安裝依賴：

```bash
uv sync --dev
```

啟動 bot：

```bash
uv run telegram-rag-chatbot
```

## Telegram 指令

```text
/start   查看簡介
/help    查看指令
/status  查看目前 ChromaDB chunks 數量
/reindex 重新索引 data/uploads 內的文件
/clear   清空向量知識庫
```

執行 `/reindex` 時，bot 會先回覆「正在重新索引，請稍等」。索引完成前如果直接提問，bot 會提醒目前正在建立索引，避免你以為它沒有反應。

## 環境變數

| 名稱 | 預設值 | 說明 |
|---|---:|---|
| `GEMINI_API_KEY` | 必填 | Gemini API key |
| `TELEGRAM_BOT_TOKEN` | 必填 | Telegram bot token |
| `DATA_DIR` | `data` | 資料目錄 |
| `CHROMA_DIR` | `data/chroma` | ChromaDB 持久化位置 |
| `UPLOAD_DIR` | `data/uploads` | Telegram 文件下載位置 |
| `COLLECTION_NAME` | `telegram_rag` | Chroma collection 名稱 |
| `GEMINI_CHAT_MODEL` | `gemini-2.5-flash` | 回答用 Gemini model |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model |
| `CHUNK_SIZE` | `1600` | 每個 chunk 的最大字元數；調大可減少 chunks、加快索引 |
| `CHUNK_OVERLAP` | `200` | chunk 之間的重疊字元數 |
| `RETRIEVAL_K` | `4` | 每次提問取回的 chunks 數量 |
| `INDEX_BATCH_SIZE` | `64` | 每批寫入 ChromaDB 的 chunks 數量 |
| `AUTO_REINDEX_ON_STARTUP` | `true` | 啟動時是否自動重建索引 |

## 可以問 Bot 的範例問題

如果你已經啟動 bot，內建範例資料會自動建進知識庫。可以直接複製下面的問題到 Telegram 測試。

### Apollo 11

```text
Apollo 11 任務的主要目標是什麼？
Apollo 11 的太空人有哪些人？
Apollo 11 任務使用了哪些主要太空載具？
Apollo 11 press kit 裡提到的任務時程有哪些重點？
請整理 Apollo 11 登月任務的重點，並附上來源。
```

### uv

```text
uv 主要解決什麼問題？
uv 和 pip 有什麼不同？
uv 支援哪些 Python 專案管理功能？
README 裡如何安裝 uv？
請用新手能懂的方式解釋 uv 的用途。
```

### Alice's Adventures in Wonderland

```text
Alice 在故事開頭遇到了什麼事件？
白兔在故事中扮演什麼角色？
Alice 掉進兔子洞後發生了哪些事？
柴郡貓有什麼特色？
請摘要 Alice's Adventures in Wonderland 的前半段劇情。
```

### Frankenstein

```text
Frankenstein 這本書的主要敘事者有哪些？
Victor Frankenstein 創造了什麼？
怪物如何描述自己的孤獨？
Walton 在故事中扮演什麼角色？
請整理 Frankenstein 的核心衝突。
```

### 混合測試

```text
目前知識庫裡有哪些範例資料來源？
請列出你找到答案時引用的來源。
如果知識庫裡沒有答案，你會怎麼回答？
請比較 Apollo 11 press kit 和 uv README 的文件類型差異。
```

## 常見問題

### Bot 沒有回應

請檢查：

- `.env` 裡的 `TELEGRAM_BOT_TOKEN` 是否正確
- 是否已經執行 `uv run telegram-rag-chatbot`
- 如果用 Docker，執行 `docker compose logs -f bot`

### Gemini API 報錯

請檢查：

- `GEMINI_API_KEY` 是否正確
- Gemini API key 是否有權限
- 是否超過 API quota

### 上傳 PDF 後沒有內容

有些 PDF 是掃描圖片，沒有可抽取文字。這個範例目前不做 OCR；請改用有文字層的 PDF，或先把內容轉成 TXT / Markdown。

### 重複上傳同一份文件會怎樣

目前會再次加入索引。若想重新整理知識庫，可以使用：

```text
/clear
/reindex
```
