run main.ipynb
順序：
1. 進入 run_parse.py
2. 執行 pdf_to_db()
    a. pdf_to_json
        - 會執行 parse_pdf.py 裡的函數，找出表和圖的位置和檔案儲存位置
        - 執行 pymupdf4llm，取得純文字文本和markdown表格樣態文字
        - 執行 preprocess_pdf() ，將 markdown 表格樣態文字轉換成 html 格式，將每一頁的文字表格和圖片進行儲存
        - 儲存每一頁對應的文字內容，以及markdown對應html的json檔案存起來（可以選擇不要存）
        - 計算各步驟的時間
    b. json_to_db
        - 讀取json
        - 進行chunking和跨頁表格合成
        - 將資料存入db，並記錄時間
3. 執行 query_and_respond()
    - 會從 db 中 query 相關資訊，並使用 GPT-4o 生成回答
    - query_and_respond(query ,collection, markdown_to_html,image_dict, k=3)
        - query: 使用者輸入的問題，只有這個是使用者自行決定
        - collection: 資料庫名稱
        - markdown_to_html: 將 markdown 轉換成 html 的函數
        - image_dict: 圖片對應的 dict
        - k: 使用多少個相關的 chunk 來生成回答