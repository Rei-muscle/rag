// ============================================================================
// LocalRAG Browser - 核心 RAG 應用程式邏輯 (ES Module)
// ============================================================================

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// 關閉本地模型載入，強制引導 Transformers.js 從 Hugging Face CDN 下載模型
env.allowLocalModels = false;

// PDF.js 核心 Worker 設定 (必須指向 CDN 以在純前端背景執行緒解析 PDF)
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js';

// ============================================================================
// 1. 全域應用程式狀態 (State)
// ============================================================================
const state = {
  // 模型配置
  modelMode: 'gemini',      // 'gemini' | 'openai' | 'local'
  apiKey: '',               // 雲端模型 API 金鑰
  
  // 文件資料
  loadedFileName: '',
  loadedFileSize: 0,
  rawText: '',
  chunks: [],               // 格式: { id, text, embedding: [...] }
  
  // Transformers.js 模型管線 (Pipelines)
  embeddingPipeline: null,  // 用於將文字轉為向量的模型
  localLLMPipeline: null,   // 本地大語言模型 (Qwen 1.5 0.5B Chat)
  
  // 運行旗標
  isProcessingFile: false,
  isGeneratingAnswer: false,
  
  // 中斷生成控制器
  abortController: null
};

// ============================================================================
// 2. 元素選取器 (DOM Elements)
// ============================================================================
const DOM = {
  // 設定面版
  modelMode: document.getElementById('model-mode'),
  apiKeyGroup: document.getElementById('api-key-group'),
  apiKeyLabel: document.getElementById('api-key-label'),
  apiKey: document.getElementById('api-key'),
  toggleKeyVisibility: document.getElementById('toggle-key-visibility'),
  apiHelp: document.getElementById('api-help'),
  saveSettingsBtn: document.getElementById('save-settings-btn'),
  
  // 文件上傳
  dropZone: document.getElementById('drop-zone'),
  fileInput: document.getElementById('file-input'),
  fileInfoContainer: document.getElementById('file-info-container'),
  loadedFileName: document.getElementById('loaded-file-name'),
  fileSizeBadge: document.getElementById('file-size-badge'),
  fileTypeBadge: document.getElementById('file-type-badge'),
  removeFileBtn: document.getElementById('remove-file-btn'),
  
  // 進度顯示與日誌
  processStatusContainer: document.getElementById('process-status-container'),
  processStatusText: document.getElementById('process-status-text'),
  processPercentage: document.getElementById('process-percentage'),
  progressBarFill: document.getElementById('progress-bar-fill'),
  logConsole: document.getElementById('log-console'),
  
  // 向量資料庫視覺化
  dbViewerTrigger: document.getElementById('db-viewer-trigger'),
  dbViewerContent: document.getElementById('db-viewer-content'),
  statsChunksCount: document.getElementById('stats-chunks-count'),
  vectorDbList: document.getElementById('vector-db-list'),
  
  // 聊天對話區
  chatMessages: document.getElementById('chat-messages'),
  chatInput: document.getElementById('chat-input'),
  sendBtn: document.getElementById('send-btn'),
  chatForm: document.getElementById('chat-form'),
  
  // 參考來源面板
  retrievalReferences: document.getElementById('retrieval-references'),
  refChunksContainer: document.getElementById('ref-chunks-container'),
  closeRefBtn: document.getElementById('close-ref-btn')
};

// ============================================================================
// 3. 系統初始化與設定保存
// ============================================================================
function init() {
  // 從 sessionStorage 載入已儲存的金鑰，模型配置依然保留在 localStorage
  const savedGeminiKey = sessionStorage.getItem('local_rag_gemini_key') || '';
  const savedOpenaiKey = sessionStorage.getItem('local_rag_openai_key') || '';
  const savedMode = localStorage.getItem('local_rag_model_mode') || 'gemini';
  
  state.modelMode = savedMode;
  DOM.modelMode.value = savedMode;
  
  // 初始化 state 中的 API Key
  if (savedMode === 'gemini') {
    state.apiKey = savedGeminiKey;
  } else if (savedMode === 'openai') {
    state.apiKey = savedOpenaiKey;
  }
  
  updateApiKeyUI(savedMode, savedGeminiKey, savedOpenaiKey);
  setupEventListeners();
  log("系統初始化完畢，準備就緒。");
}

// 根據選擇的模式更新金鑰輸入 UI
function updateApiKeyUI(mode, geminiKey = '', openaiKey = '') {
  if (mode === 'local') {
    DOM.apiKeyGroup.classList.add('hidden');
  } else {
    DOM.apiKeyGroup.classList.remove('hidden');
    if (mode === 'gemini') {
      DOM.apiKeyLabel.textContent = 'Gemini API 金鑰';
      DOM.apiKey.placeholder = '請輸入 Gemini API 金鑰 (AI Studio)...';
      DOM.apiKey.value = geminiKey || sessionStorage.getItem('local_rag_gemini_key') || '';
      DOM.apiHelp.innerHTML = `
        金鑰僅暫存於此分頁 (Session)，關閉網頁即自動刪除。
        <a href="https://aistudio.google.com/" target="_blank" rel="noopener noreferrer" class="link-inline">點此申請免費 Gemini 金鑰</a>
      `;
    } else if (mode === 'openai') {
      DOM.apiKeyLabel.textContent = 'OpenAI API 金鑰';
      DOM.apiKey.placeholder = '請輸入 OpenAI API 金鑰 (sk-...)...';
      DOM.apiKey.value = openaiKey || sessionStorage.getItem('local_rag_openai_key') || '';
      DOM.apiHelp.innerHTML = `
        金鑰僅暫存於此分頁 (Session)，關閉網頁即自動刪除。
        <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" class="link-inline">點此獲取 OpenAI 金鑰</a>
      `;
    }
  }
}

// 寫入運行日誌至網頁底部的 Console
function log(message, isError = false) {
  const line = document.createElement('div');
  line.className = `log-line ${isError ? 'error' : ''}`;
  const time = new Date().toLocaleTimeString();
  line.textContent = `[${time}] ${message}`;
  DOM.logConsole.appendChild(line);
  DOM.logConsole.scrollTop = DOM.logConsole.scrollHeight;
  console.log(`[RAG-Log] ${message}`);
}

// ============================================================================
// 4. 事件監聽 (Event Listeners)
// ============================================================================
function setupEventListeners() {
  // 切換推理引擎下拉選單
  DOM.modelMode.addEventListener('change', (e) => {
    updateApiKeyUI(e.target.value);
  });

  // 切換 API 金鑰顯示/隱藏
  DOM.toggleKeyVisibility.addEventListener('click', () => {
    const icon = DOM.toggleKeyVisibility.querySelector('span');
    if (DOM.apiKey.type === 'password') {
      DOM.apiKey.type = 'text';
      icon.textContent = 'visibility_off';
    } else {
      DOM.apiKey.type = 'password';
      icon.textContent = 'visibility';
    }
  });

  // 保存設定按鈕
  DOM.saveSettingsBtn.addEventListener('click', () => {
    const mode = DOM.modelMode.value;
    const keyValue = DOM.apiKey.value.trim();
    
    state.modelMode = mode;
    localStorage.setItem('local_rag_model_mode', mode);
    
    if (mode === 'gemini') {
      sessionStorage.setItem('local_rag_gemini_key', keyValue);
      sessionStorage.removeItem('local_rag_openai_key'); // 清除另一種金鑰
      state.apiKey = keyValue;
      log("Gemini API 金鑰已安全地暫存於本分頁。");
    } else if (mode === 'openai') {
      sessionStorage.setItem('local_rag_openai_key', keyValue);
      sessionStorage.removeItem('local_rag_gemini_key'); // 清除另一種金鑰
      state.apiKey = keyValue;
      log("OpenAI API 金鑰已安全地暫存於本分頁。");
    } else {
      log("已切換為純本地 AI 運作模式。");
    }
    
    alert("系統配置已成功儲存！(金鑰將在關閉分頁後自動清除)");
  });

  // 折疊式向量資料庫標題點擊
  DOM.dbViewerTrigger.addEventListener('click', () => {
    DOM.dbViewerTrigger.classList.toggle('collapsed');
    DOM.dbViewerContent.classList.toggle('collapsed');
  });

  // 關閉檢索來源展示
  DOM.closeRefBtn.addEventListener('click', () => {
    DOM.retrievalReferences.classList.add('hidden');
  });

  // 拖曳文件互動
  DOM.dropZone.addEventListener('click', () => DOM.fileInput.click());
  
  DOM.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    DOM.dropZone.classList.add('dragover');
  });

  // 雙面板拖曳調整高度邏輯
  const resizeHandle = document.getElementById('panel-resize-handle');
  const uploadCard = document.querySelector('.upload-card');
  
  if (resizeHandle && uploadCard) {
    let startY = 0;
    let startHeight = 0;
    
    function onPointerMove(e) {
      const deltaY = e.clientY - startY;
      const newHeight = Math.max(80, startHeight + deltaY);
      uploadCard.style.height = `${newHeight}px`;
    }
    
    function onPointerUp() {
      resizeHandle.classList.remove('dragging');
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerup', onPointerUp);
    }
    
    resizeHandle.addEventListener('pointerdown', (e) => {
      startY = e.clientY;
      startHeight = uploadCard.getBoundingClientRect().height;
      resizeHandle.classList.add('dragging');
      document.addEventListener('pointermove', onPointerMove);
      document.addEventListener('pointerup', onPointerUp);
      e.preventDefault();
    });
  }
  DOM.dropZone.addEventListener('dragleave', () => {
    DOM.dropZone.classList.remove('dragover');
  });

  DOM.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    DOM.dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  DOM.fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  // 移除載入的文件
  DOM.removeFileBtn.addEventListener('click', () => {
    resetDocumentState();
    log("文件已移除，知識庫重設。");
  });

  // 聊天送出表單
  DOM.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleUserQuery();
  });
}

// 重設文件狀態與 UI
function resetDocumentState() {
  state.loadedFileName = '';
  state.loadedFileSize = 0;
  state.rawText = '';
  state.chunks = [];
  
  // 清空檔案輸入框的值，允許重複選擇同一個檔案
  if (DOM.fileInput) {
    DOM.fileInput.value = '';
  }
  
  DOM.fileInfoContainer.classList.add('hidden');
  DOM.processStatusContainer.classList.add('hidden');
  DOM.dropZone.classList.remove('hidden');
  
  DOM.chatInput.disabled = true;
  DOM.chatInput.placeholder = "請先載入文件，即可在此提問文件內容...";
  DOM.sendBtn.disabled = true;
  
  DOM.statsChunksCount.textContent = '0';
  DOM.vectorDbList.innerHTML = '<div class="db-empty-state">尚未載入任何文件數據</div>';
  
  DOM.retrievalReferences.classList.add('hidden');
}

// ============================================================================
// 5. 文件處理與文字解析 (TXT & PDF)
// ============================================================================
async function handleFileSelected(file) {
  if (state.isProcessingFile) return;
  
  const ext = file.name.split('.').pop().toLowerCase();
  if (ext !== 'pdf' && ext !== 'txt') {
    alert("僅支援上傳 PDF 或 TXT 格式檔案。");
    return;
  }

  // 顯示載入中 UI
  state.isProcessingFile = true;
  DOM.dropZone.classList.add('hidden');
  DOM.fileInfoContainer.classList.remove('hidden');
  DOM.loadedFileName.textContent = file.name;
  DOM.fileSizeBadge.textContent = `${Math.round(file.size / 1024)} KB`;
  DOM.fileTypeBadge.textContent = ext.toUpperCase();
  
  DOM.processStatusContainer.classList.remove('hidden');
  updateProgress(0, "讀取檔案中...");

  try {
    let text = "";
    if (ext === 'txt') {
      text = await readTextFile(file);
    } else if (ext === 'pdf') {
      text = await readPDFFile(file);
    }

    if (!text.trim()) {
      throw new Error("無法從檔案中提取出任何文字內容。");
    }

    state.rawText = text;
    log(`文件解析成功。字數: ${text.length}`);
    
    // 執行文本切片與向量化
    await processDocumentRAG(text);

  } catch (error) {
    log(`檔案處理失敗: ${error.message}`, true);
    alert(`讀取檔案出錯: ${error.message}`);
    state.isProcessingFile = false; // 優先強制設為 false
    resetDocumentState();
  } finally {
    state.isProcessingFile = false;
  }
}

// 讀取純文字檔
function readTextFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (err) => reject(err);
    reader.readAsText(file, 'UTF-8');
  });
}

// 使用 PDF.js 進行本機 PDF 解析
async function readPDFFile(file) {
  log("正在使用 PDF.js 提取本機 PDF 文字...");
  const arrayBuffer = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (err) => reject(err);
    reader.readAsArrayBuffer(file);
  });

  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const numPages = pdf.numPages;
  let text = "";
  
  for (let i = 1; i <= numPages; i++) {
    updateProgress(Math.round((i / numPages) * 30), `解析 PDF 文字 (頁數: ${i}/${numPages})...`);
    const page = await pdf.getPage(i);
    const textContent = await page.getTextContent();
    const strings = textContent.items.map(item => item.str);
    text += strings.join(" ") + "\n";
  }
  return text;
}

// 更新進度條 UI
function updateProgress(percentage, text) {
  DOM.processStatusText.textContent = text;
  DOM.processPercentage.textContent = `${percentage}%`;
  DOM.progressBarFill.style.width = `${percentage}%`;
}

// ============================================================================
// 6. RAG 段落切片 (Chunking) 演算法
// ============================================================================
/**
 * 將超長文本以「滑動視窗字元重疊法」進行切片
 * 這樣可以避免句子或段落的語意被強行截斷
 */
function chunkDocument(text, chunkSize = 400, overlap = 80) {
  // 將連續多個空白與換行整理為單一空白，以利語意向量模型判斷
  const cleanText = text.replace(/\s+/g, ' ').trim();
  const chunks = [];
  let index = 0;
  
  while (index < cleanText.length) {
    let end = Math.min(index + chunkSize, cleanText.length);
    
    // 試圖在標點符號或空格處截斷，保持句法結構完整
    if (end < cleanText.length) {
      const boundaryMarkers = ['。', '！', '？', '；', '\n', ' ', '.', '!', '?'];
      let bestCut = -1;
      
      // 在後 20% 的視窗中尋找適合的句子結束標記
      const searchRange = Math.round(chunkSize * 0.25);
      const searchStart = end - searchRange;
      
      for (let j = end; j >= searchStart; j--) {
        if (boundaryMarkers.includes(cleanText[j])) {
          bestCut = j + 1; // 包含該結尾標籤
          break;
        }
      }
      if (bestCut !== -1) {
        end = bestCut;
      }
    }
    
    const chunkText = cleanText.slice(index, end).trim();
    if (chunkText.length > 10) { // 過濾過短的雜訊分段
      chunks.push({
        id: chunks.length + 1,
        text: chunkText,
        embedding: null
      });
    }
    
    // 往後滑動 (扣除重疊字數)
    index += (chunkSize - overlap);
  }
  return chunks;
}

// ============================================================================
// 7. 本地向量庫建置 (Transformers.js 嵌入)
// ============================================================================
async function processDocumentRAG(text) {
  updateProgress(35, "規劃段落切片...");
  const rawChunks = chunkDocument(text);
  log(`文件段落切片完成。共切為 ${rawChunks.length} 個段落。`);
  
  // 載入本地向量嵌入模型
  updateProgress(40, "正在加載嵌入模型 (MiniLM)...");
  let extractor;
  try {
    extractor = await getEmbeddingPipeline((progress, statusText) => {
      // 下載模型的進度對應到 UI 的 40% - 60%
      const currentPercentage = 40 + Math.round(progress * 20);
      updateProgress(currentPercentage, statusText);
    });
  } catch (err) {
    throw new Error(`載入本地嵌入模型失敗: ${err.message}`);
  }

  // 對每個段落進行向量化 (Embedding)
  updateProgress(65, "計算語意向量值...");
  
  const total = rawChunks.length;
  for (let i = 0; i < total; i++) {
    const chunk = rawChunks[i];
    const progressPercent = 65 + Math.round((i / total) * 30);
    updateProgress(progressPercent, `產生語意向量 (${i + 1}/${total})...`);
    
    // 生成特徵向量 (以 mean pooling 與 L2 向量正規化處理)
    const embedding = await generateEmbedding(chunk.text, extractor);
    chunk.embedding = embedding;
  }

  state.chunks = rawChunks;
  
  // 更新 UI 狀態
  updateProgress(100, "資料庫建置完成！");
  log("本地向量資料庫建置成功。");
  
  // 啟用輸入欄位與視覺化面板
  DOM.chatInput.disabled = false;
  DOM.chatInput.placeholder = "請輸入關於文件內容的提問...";
  DOM.sendBtn.disabled = false;
  
  DOM.statsChunksCount.textContent = total;
  renderVectorDatabase();
}

// 取得或快取 Embedding Pipeline
async function getEmbeddingPipeline(progressCallback) {
  if (!state.embeddingPipeline) {
    state.embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      progress_callback: (p) => {
        if (p.status === 'progress') {
          progressCallback(p.progress, `下載語意嵌入模型: ${Math.round(p.progress)}%`);
        }
      }
    });
  }
  return state.embeddingPipeline;
}

// 計算文字的 Embedding 數值
async function generateEmbedding(text, extractor) {
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  // 將 Tensor 轉為 JavaScript 普通陣列
  return Array.from(output.data);
}

// 視覺化渲染左側的「向量資料庫展示」 (RAG 科普/專題加分器)
function renderVectorDatabase() {
  DOM.vectorDbList.innerHTML = '';
  
  state.chunks.forEach(chunk => {
    const item = document.createElement('div');
    item.className = 'vector-chunk-item';
    
    // 取前五個向量數值展示，後續以省略號表示
    const vectorPreview = `[${chunk.embedding.slice(0, 5).map(v => v.toFixed(4)).join(', ')}, ...]`;
    
    item.innerHTML = `
      <div class="chunk-header-info">
        <span># 段落 ${chunk.id}</span>
        <span>${chunk.embedding.length} 維向量</span>
      </div>
      <div class="chunk-text-preview">${escapeHtml(chunk.text)}</div>
      <div class="vector-preview-values" title="完整向量長度: ${chunk.embedding.length}">${vectorPreview}</div>
    `;
    DOM.vectorDbList.appendChild(item);
  });
}

// HTML 安全編碼防止注入
function escapeHtml(string) {
  return String(string)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ============================================================================
// 8. 餘弦相似度 (Cosine Similarity) 檢索演算法
// ============================================================================
/**
 * 計算兩向量間之夾角餘弦值
 * 數值落於 [-1, 1]，通常在 RAG 中 0.5 ~ 1 表示有語意相關性，越接近 1 越相關
 */
function calculateCosineSimilarity(vecA, vecB) {
  let dotProduct = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// 根據使用者問題進行 RAG 語意搜尋
async function performVectorRetrieval(queryText, topK = 3) {
  log(`進行向量語意搜尋。問題: "${queryText}"`);
  
  // 1. 生成問題本身的向量值
  const extractor = await getEmbeddingPipeline();
  const queryEmbedding = await generateEmbedding(queryText, extractor);
  
  // 2. 計算問題向量與庫中所有段落向量的相似度
  const scoredChunks = state.chunks.map(chunk => {
    const similarity = calculateCosineSimilarity(queryEmbedding, chunk.embedding);
    return {
      ...chunk,
      similarity: similarity
    };
  });
  
  // 3. 排序並取出前 TopK 名最相關段落
  scoredChunks.sort((a, b) => b.similarity - a.similarity);
  return scoredChunks.slice(0, topK);
}

// ============================================================================
// 9. 對話流程與 LLM 生成介接
// ============================================================================
async function handleUserQuery() {
  // 如果目前正在生成，此時點擊按鈕代表「停止/中斷生成」
  if (state.isGeneratingAnswer) {
    if (state.abortController) {
      log("使用者手動中斷了 AI 答案生成。");
      state.abortController.abort();
    }
    return;
  }

  const queryText = DOM.chatInput.value.trim();
  if (!queryText) return;
  
  // 檢查 API 金鑰 (在雲端模式下)
  if (state.modelMode !== 'local') {
    const key = DOM.apiKey.value.trim();
    if (!key) {
      alert("使用雲端模型請先在左側輸入並保存 API 金鑰。");
      return;
    }
    state.apiKey = key;
  }

  // 1. 清空輸入並將使用者訊息渲染至對話區
  DOM.chatInput.value = '';
  appendMessage('user', queryText);
  
  // 2. 顯示 AI 思考與載入指示器，並更新發送按鈕為「停止按鈕」
  const aiMessageElement = appendMessage('ai', '', true);
  state.isGeneratingAnswer = true;
  state.abortController = new AbortController();
  
  // 切換發送按鈕為「中斷/停止」狀態
  DOM.sendBtn.disabled = false;
  DOM.sendBtn.classList.add('btn-stop');
  DOM.sendBtn.title = "停止生成";
  DOM.sendBtn.innerHTML = `<span class="material-symbols-outlined">stop</span>`;
  
  try {
    // 3. 執行 RAG：語意檢索 Top-3 段落
    const matchedChunks = await performVectorRetrieval(queryText, 3);
    log(`檢索到 ${matchedChunks.length} 個最相關段落。最高相似度: ${Math.round(matchedChunks[0].similarity * 100)}%`);
    
    // 顯示即時檢索結果面板
    displayRetrievalReferences(matchedChunks);

    // 4. 將檢索出的段落作為 Context 組裝 Prompt
    const contextText = matchedChunks.map(c => `[資料來源段落 ${c.id}] (相似度: ${Math.round(c.similarity * 100)}%):\n${c.text}`).join('\n\n');
    
    // 5. 呼叫對應 LLM 生成模型回答，傳入 abort signal
    let answerText = "";
    if (state.modelMode === 'gemini') {
      answerText = await generateWithGemini(queryText, contextText, state.abortController.signal);
    } else if (state.modelMode === 'openai') {
      answerText = await generateWithOpenAI(queryText, contextText, state.abortController.signal);
    } else if (state.modelMode === 'local') {
      answerText = await generateWithLocalLLM(queryText, contextText, state.abortController.signal);
    }

    // 6. 移除思考動畫，並填入真正的 AI 回答
    const typingIndicator = aiMessageElement.querySelector('.ai-typing-indicator');
    if (typingIndicator) typingIndicator.remove();
    
    const bubble = aiMessageElement.querySelector('.message-bubble');
    bubble.innerHTML = formatMarkdownAnswer(answerText);
    
    // 7. 在對話卡片下方附上檢索來源的按鈕，方便報告查看
    appendSourcesToggle(bubble, matchedChunks);
    
  } catch (error) {
    const isAborted = error.name === 'AbortError' || error.message.includes('aborted');
    const displayMsg = isAborted ? "（已由使用者停止生成）" : `抱歉，在計算答案時發生錯誤: ${error.message}`;
    log(isAborted ? "生成已被使用者停止。" : `回答生成出錯: ${error.message}`, !isAborted);
    
    const typingIndicator = aiMessageElement.querySelector('.ai-typing-indicator');
    if (typingIndicator) typingIndicator.remove();
    
    const bubble = aiMessageElement.querySelector('.message-bubble');
    bubble.innerHTML = `<p style="${isAborted ? 'color:var(--text-muted);font-style:italic;' : 'color:#ef4444;'}">${displayMsg}</p>`;
  } finally {
    state.isGeneratingAnswer = false;
    state.abortController = null;
    
    // 恢復發送按鈕樣式
    DOM.sendBtn.classList.remove('btn-stop');
    DOM.sendBtn.title = "發送訊息";
    DOM.sendBtn.innerHTML = `<span class="material-symbols-outlined">send</span>`;
    
    // 依據輸入框是否可用來重設按鈕可用性
    if (DOM.chatInput.value.trim() === '' && DOM.chatInput.disabled) {
      DOM.sendBtn.disabled = true;
    }
    
    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
  }
}

// 渲染單條聊天訊息
function appendMessage(role, text, isPending = false) {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;
  
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  const icon = role === 'user' ? 'person' : (role === 'system' ? 'smart_toy' : 'robot_2');
  avatar.innerHTML = `<span class="material-symbols-outlined">${icon}</span>`;
  
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  
  if (isPending) {
    bubble.innerHTML = `
      <div class="ai-typing-indicator">
        <span></span>
        <span></span>
        <span></span>
      </div>
    `;
  } else {
    bubble.innerHTML = `<p>${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;
  }
  
  msg.appendChild(avatar);
  msg.appendChild(bubble);
  DOM.chatMessages.appendChild(msg);
  DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
  
  return msg;
}

// 展示下方即時的語意檢索參考來源面板
function displayRetrievalReferences(chunks) {
  DOM.refChunksContainer.innerHTML = '';
  DOM.retrievalReferences.classList.remove('hidden');
  
  chunks.forEach((chunk, i) => {
    const item = document.createElement('div');
    item.className = 'ref-item';
    item.innerHTML = `
      <div class="ref-item-title">
        <span># 段落 ${chunk.id}</span>
        <span class="ref-similarity">${Math.round(chunk.similarity * 100)}% 相似</span>
      </div>
      <div class="ref-item-text">${escapeHtml(chunk.text)}</div>
    `;
    DOM.refChunksContainer.appendChild(item);
  });
}

// 將對話泡泡內文進行簡易的 Markdown 格式轉換 (如 **粗體**、條列項目、斜體)
function formatMarkdownAnswer(text) {
  if (!text) return '';

  let html = escapeHtml(text);
  html = html
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/~~(.+?)~~/g, '<del>$1</del>')
    .replace(/`([^`]+?)`/g, '<code>$1</code>');

  html = html.replace(/\r\n/g, '\n').replace(/\n{2,}/g, '</p><p>');
  html = `<p>${html}</p>`;
  html = html.replace(/\n/g, '<br>');

  return html;
}

function appendSourcesToggle(bubble, chunks) {
  if (!chunks || chunks.length === 0) return;

  const details = document.createElement('details');
  details.className = 'source-toggle';
  details.innerHTML = `
    <summary>顯示檢索來源</summary>
    <div class="source-list">
      ${chunks.map(chunk => `
        <div class="source-item">
          <div class="source-title">段落 ${chunk.id} · 相似度 ${Math.round(chunk.similarity * 100)}%</div>
          <div class="source-text">${escapeHtml(chunk.text)}</div>
        </div>
      `).join('')}
    </div>
  `;

  bubble.appendChild(details);
}

// 模式 C: 純本地大模型 (Transformers.js + Qwen 1.5 0.5B Chat)
async function generateWithLocalLLM(query, context, signal) {
  log("正在準備本地 Qwen 大模型進行推理 (這將 100% 消耗您的本機 CPU/GPU)...");
  
  // 初始化本地大模型管線 (首次加載時會在 UI 進度條呈現)
  const generator = await getLocalLLMPipeline((progress, statusText) => {
    // 將大模型下載進度展示在 Console 中，提示使用者
    log(`[模型加載中] ${statusText}`);
  });

  // 使用 Qwen 的 Chat Template 組裝對話格式
  const prompt = `<|im_start|>system
你是一個文件分析助理。請嚴格根據以下提供的文件段落回答問題。如果無法回答，請說不知道。
給定文件段落：
${context}<|im_end|>
<|im_start|>user
${query}<|im_end|>
<|im_start|>assistant
`;

  log("本地模型正在編譯上下文並進行推理生成答案...");
  
  const output = await generator(prompt, {
    max_new_tokens: 256,
    temperature: 0.2,
    do_sample: false,
    return_full_text: false,
    callback_function: () => {
      if (signal && signal.aborted) {
        throw new Error('aborted');
      }
    }
  });

  let generatedText = output[0]?.generated_text || "";
  generatedText = generatedText.replace(prompt, "").trim();
  generatedText = generatedText.split("<|im_end|>")[0].trim();

  if (!generatedText) {
    generatedText = "（本地模型未能生成答案，可能因為內容過長或硬體算力受限，建議切換為 Gemini 雲端模型）";
  }

  log("本地大模型推理完成。");
  return generatedText;
}

// 模式 A: 呼叫 Google Gemini API (3.1 Flash-Lite ── 小巧高效，免費層有充裕配額)
async function generateWithGemini(query, context, signal) {
  log("發送請求至 Google Gemini API...");
  
  const prompt = `
你是一個基於給定文件內容進行回答的專業AI助理。請嚴格根據以下提供的文件段落回答問題。
如果給定的文件段落無法回答該問題，請回答「對不起，根據載入的文件內容，無法提供相關解答。」，不要憑空捏造。

【給定的文件段落】：
${context}

【使用者的問題】：
${query}

請提供條理清晰、切合問題的回答：
`;

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key=${state.apiKey}`;
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      contents: [{
        parts: [{ text: prompt }]
      }],
      generationConfig: {
        temperature: 0.2,
        maxOutputTokens: 1000
      }
    }),
    signal: signal
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    const errMsg = errData.error?.message || `HTTP code ${response.status}`;
    throw new Error(`Gemini API 呼叫失敗: ${errMsg}`);
  }

  const data = await response.json();
  const answer = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!answer) {
    throw new Error("Gemini API 未回傳有效的文本內容。");
  }
  return answer;
}

// 模式 B: 呼叫 OpenAI API
async function generateWithOpenAI(query, context, signal) {
  log("發送請求至 OpenAI API...");
  
  const systemPrompt = "你是一個文件分析助理。請僅根據提供的文件內容回答問題。如果無法從文件回答，請老實回答「對不起，根據載入的文件內容，無法提供相關解答。」。";
  const userPrompt = `
【參考文件段落】：
${context}

【問題】：
${query}
`;

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${state.apiKey}`
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      temperature: 0.2
    }),
    signal: signal
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    const errMsg = errData.error?.message || `HTTP code ${response.status}`;
    throw new Error(`OpenAI API 呼叫失敗: ${errMsg}`);
  }

  const data = await response.json();
  const answer = data.choices?.[0]?.message?.content;
  if (!answer) {
    throw new Error("OpenAI API 未回傳有效回答。");
  }
  return answer;
}

// 模式 C: 純本地大模型 (Transformers.js + Qwen 1.5 0.5B Chat)
async function getLocalLLMPipeline(progressCallback) {
  if (!state.localLLMPipeline) {
    log("開始下載並解析 Qwen 1.5 Chat 0.5B 模型 (約 350MB, 請保持網路暢通)...");
    
    state.localLLMPipeline = await pipeline('text-generation', 'Xenova/Qwen1.5-0.5B-Chat', {
      progress_callback: (p) => {
        if (p.status === 'progress') {
          progressCallback(p.progress, `正在載入本地語言模型: ${Math.round(p.progress)}%`);
        }
      }
    });
    log("本地語言模型快取與載入成功。");
  }
  return state.localLLMPipeline;
}

// ============================================================================
// 11. 啟動應用程式
// ============================================================================
window.addEventListener('DOMContentLoaded', init);
