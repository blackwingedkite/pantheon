import re
import os
from GeneralAgent import Agent
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

role_prompt = "你現在是一個專業的機器人，可以針對HTML內容進行分析，並且提供相關的建議。"
whether_to_merge_prompt = """
Carefully examine the two provided tables and determine whether they should be merged into a single table.
If you believe they should be merged, answer "True". If not, answer "False". Do not output any other text.

Consider the following:
1. If the tables have partially similar content but some differences, yet are about the same topic, output "True".
2. If the tables have completely different content or are about different topics, output "False".
3. Observe the table titles, content, and format to make your decision.
4. If the first table has a title but the second doesn't, there's a higher chance they're part of the same table.
5. Look for <thead> and </thead> tags to aid your judgment.
6. Consider the feasibility and difficulty of merging when making your decision. This point is quite important because you will be asked to merge the tables if you answer "True".

Your decision will determine whether we proceed with merging (if "True") or keep the tables separate (if "False").
This decision is crucial and it very important to my career. 

Please choose carefully between "True" and "False" only, without any additional text.

"""
merging_prompt = """
**Instructions for Merging Two HTML Tables into One Coherent, Valid HTML Table**

### Important Notes:

1. **Table Title**: If one table has a title and the other doesn’t, use the existing title as the final table's title. Ensure the title is clear, accurate, and prominent for readability.
2. **Unique Data**: Combine the content from both tables, preserving all unique information from each.
3. **Remove Redundancy**: Eliminate unused or redundant content to keep the final table concise and readable.
4. **Reconcile Structures**: If the tables have different column structures, merge them logically. Ensure each column serves a clear purpose.
5. **HTML Validity**: Ensure the final table is well-formatted, semantically correct, and fully HTML valid.
6. **Semantic Structure**: Properly use `<thead>`, `<tbody>`, and `<tfoot>` tags to reflect the structure of the merged content.
7. You can add cells if needed. BUT you should always try to better the content of the output table.

### Final Deliverable:
- Provide the HTML code for the merged table, formatted and indented for readability.
- YOU SHOULD ALWAYS USE THE FIRST TABLE'S TITLE AS THE MERGED TABLE'S TITLE. AND YOU SHOULD CONSIDER THE SECOND TABLE'S CONTENT TO BE MERGED INTO THE FIRST TABLE WITH THE FIRST TABLE'S CONTENT.
- THE LOGIC OF THE SECOND TABLE SHOULD BE THE SAME AS THE FIRST TABLE.
- don't response anything not html code, like "### Key Points:", "Explanation", and so on .
"""

# merging_prompt = """
# 你現在是一個表格合併的專家，你將會在受監督的情況下合併兩個 HTML 表格為一個連貫且有效的 HTML 表格的指示。

# ### 重要注意事項：

# 1. **表格標題**：如果其中一個表格有標題而另一個沒有，請使用已有的標題作為最終表格的標題。確保標題清晰、準確且醒目，以便提高可讀性。
#     － 如果兩個表格都有標題，請使用第一個表格的標題作為最終表格的標題。
# 2. **獨特資料**：合併兩個表格的內容，保留每個表格中的所有獨特信息。
# 3. **移除冗餘**：刪除未使用或冗餘的內容，保持最終表格簡潔易讀。
# 4. **結構調整**：如果兩個表格的欄位結構不同，請邏輯地將它們合併。確保每個欄位有明確的用途。
# 5. **HTML 合法性**：確保最終表格格式良好、語義正確，並保持完整的 HTML 合法性。
# 6. **語義結構**：正確使用 `<thead>`、`<tbody>` 和 `<tfoot>` 標籤，反映合併後內容的結構。
# 7. 你可以在必要時新增單元格，但應盡量改善輸出表格的內容。

# 在輸出之前，你應該要完整看過你所輸出的表格內容，確認你是否能夠讀懂。

# ### 輸出前請再注意以下內容
# - 提供合併後表格的 HTML 代碼，格式化並縮排以提高可讀性。
# - 你應該始終使用第一個表格的標題作為合併後的標題，並將第二個表格的內容合併到第一個表格的結構中，並保持邏輯一致
# - 
# - 不要回應任何非 HTML 代碼的內容，例如"### Key Points:", "Explanation", and so on .

# """

def gpt_should_merge(upper_table, lower_table, model):
    agent = Agent(role=role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
    local_prompt = whether_to_merge_prompt + "Table 1:\n" + upper_table+ "Table 1:\n" + lower_table
    content = agent.run([local_prompt]) # , {'image': output_dir+table_path}可以考慮放原始照片
    print("GPT decision is: ", content)
    if "True" in content or "true" in content:
        return True
    else:
        return False

def gpt_start_merge(upper_table, lower_table, model):
    agent = Agent(role=role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
    local_prompt = merging_prompt + "Table 1(first table):\n" + upper_table+ "Table 2(second table):\n" + lower_table
    content = agent.run([local_prompt]) # , {'image': output_dir+table_path}可以考慮放原始照片
    return content
    #return upper_table+"\n\n以上是uppertable, 以下是lowertable\n\n"+lower_table
def normalize_page(page_info):
    if page_info["start_page"] == "Not Found":
        page_info["start_page"] = 999999999
        page_info["end_page"] = -1
    return page_info

def clean_string(text):
    # 移除markdown圖片, 超連結
    # pattern =  r'\[([^\]]+)\]\(([^\)]+)\)'
    # text = re.sub(pattern, '', text)
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(pattern, '', text)
    # 移除換行符號
    text = text.replace('\n', '')
    return text

def find_unique_table(text, pattern):
    markdown_content = []
    for match in pattern.finditer(text):
        markdown_content.append((match.start(), match.end(), match.group()))
    markdown_content.sort(key=lambda x: x[0])
    # 用於存儲唯一的表格信息
    matches = []
    
    for i, content in enumerate(markdown_content):
        is_unique = True
        for j, other_content in enumerate(markdown_content):
            if i != j and content[0] >= other_content[0] and content[1] < other_content[1]:
                # 這個表格是另一個表格的子集，所以它不是唯一的
                is_unique = False
                break
        if is_unique:
            matches.append(content)
    return  matches


def chunking(
    txt_pages_dict,
    markdown_to_gptcontent, 
    model = "gpt-4o-mini", 
    chunk_size=500,
    merge_tables = True):
    text = "".join([page["text"] for page in txt_pages_dict])

    pattern = re.compile(
        r'(\|(?:[^\n]*\|)+\n'   # 匹配表格头部行
        r'\|(?:\s*[-:]+\s*\|)+\s*\n'  # 匹配表格分隔行
        r'(?:\|(?:[^\n]*\|)\n)+)',  # 匹配表格内容行
        re.DOTALL | re.MULTILINE
        )
    matches = find_unique_table(text, pattern)

    if merge_tables:
        merge_flag_list = []
        # Step 1:檢查這筆資料和下筆資料的距離，match[-1]=是否該檢查
        for index, current_item in enumerate(matches):
            # 檢查與下一個區間的距離
            if index >= len(matches) - 1:
                is_close = False  # 最後一筆數據永遠是 False
            else:
                next_item = matches[index+1]
                distance_to_next = next_item[0] - current_item[1]
                is_close = distance_to_next < 80
            original_tuple = matches[index]
            add_bool = (is_close,)
            new_tuple = original_tuple + add_bool
            merge_flag_list.append(new_tuple)
        
        
            # Step 2: match[-1]=是否該合併（但合併就是用html表格來合併）
            for index, current_item in enumerate(merge_flag_list):
                if (current_item[-1] == True) and (index != len(merge_flag_list) - 1):
                    upper_tuple = merge_flag_list[index]
                    lower_tuple = merge_flag_list[index+1]
                    gpt_judge = gpt_should_merge(upper_tuple[2], lower_tuple[2],model)
                    temp_list = []
                    for iteem in current_item:
                        temp_list.append(iteem)
                    temp_list[3] = gpt_judge
                    merge_flag_list[index] = tuple(temp_list)

            # Step 3:如果要合併，則開始合併並且維持markdown to html的對應dictionary
            for index, current_item in enumerate(merge_flag_list):

                if (current_item[-1] == True) and (index != len(merge_flag_list) - 1):
                    upper_tuple = merge_flag_list[index]
                    lower_tuple = merge_flag_list[index+1]
                    upper_html = markdown_to_gptcontent[upper_tuple[2]]
                    lower_html = markdown_to_gptcontent[lower_tuple[2]]
                    merged_content = gpt_start_merge(upper_html, lower_html, model)
                    markdown_to_gptcontent[upper_tuple[2]] = merged_content
                    markdown_to_gptcontent[lower_tuple[2]] = "Has been merged with last table"


    for md, content in enumerate(markdown_to_gptcontent):
        last_gpt_content = ""
        if content != "Has been merged with last table":
            last_gpt_content = content
        else:
            markdown_to_gptcontent[md] = last_gpt_content
            continue





    chunk_start = 0
    chunk_end = 0
    match_item = 0
    chunks = []
    start_checkpoint = []
    end_checkpoint = []
    for i in matches:
        start_checkpoint.append(i[0])
        end_checkpoint.append(i[1])
    while True:
        if match_item < len(matches):
            if chunk_start + chunk_size >= len(text):
                print(str(chunk_start)+ " "+str(len(text)) )
                chunks.append(text[chunk_start:])
                break
            elif chunk_start + chunk_size <= start_checkpoint[match_item]:
                chunk_end = chunk_start + chunk_size
                print(str(chunk_start)+ " "+str(chunk_end))
                chunks.append(text[chunk_start:chunk_end])
                chunk_start = chunk_end
            elif chunk_start + chunk_size > start_checkpoint[match_item]:
                chunks.append(text[chunk_start:start_checkpoint[match_item]])
                print(str(chunk_start)+ " "+str(start_checkpoint[match_item]))
                chunks.append(text[start_checkpoint[match_item]:end_checkpoint[match_item]])
                print(str(start_checkpoint[match_item])+ " "+str(end_checkpoint[match_item]))
                chunk_start = end_checkpoint[match_item]
                chunk_end = end_checkpoint[match_item]
                match_item += 1
        else:
            if chunk_start + chunk_size >= len(text):
                print(str(chunk_start)+ " "+str(len(text)) )
                chunks.append(text[chunk_start:])
                break
            else:
                chunk_end = chunk_start + chunk_size
                print(str(chunk_start)+ " "+str(chunk_end))
                chunks.append(text[chunk_start:chunk_end])
                chunk_start = chunk_end

    #標注頁碼
    now_page = 0
    chunk_page = list()
    print("Number of chunks:", len(chunks))
    print("Number of pages in txt_pages_dict:", len(txt_pages_dict))
    for chunk in chunks:
        inner_dict = {"chunk": chunk, "start_page": "", "end_page":""}

        #first_content = 有可能的第一頁的內容
        #last_content = 有可能的最後一頁的內容
        #content = 這之間的內容（可能跨多頁）
        first_content = txt_pages_dict[now_page]["text"]
        content = first_content
        last_content = ""
        forward_step = 0
        page_loc = []
        while now_page < len(txt_pages_dict): #從第一頁到最後一頁
            if (chunk in last_content) and (chunk not in first_content):
                #所有的內容不在原本這頁了，但是在下一頁．這之後就完全忽略前一頁的內容
                now_page += 1
                page_loc.append(now_page +1)
                break
            elif (chunk in content):
                #跨多頁的內容（含第一頁到最後一頁）會在這裡處理
                end_page = now_page + forward_step
                for step in range(now_page, end_page+1):
                    page_loc.append(step+1)
                if last_content != "":
                    now_page = now_page + forward_step # - 1
                break # 找到 chunk 所在的頁面後退出循環
            else:
                #如果在目前的content找不到，就往下一頁找
                forward_step += 1
                #如果已經到最後一頁了，就不再往下找
                if now_page + forward_step >= len(txt_pages_dict):
                    break
                last_content = txt_pages_dict[now_page+forward_step]["text"]
                content += last_content
        
        if len(page_loc) == 0:
            print("Not Found")
            page_txt = "Not Found"
            inner_dict["start_page"] = page_txt
            inner_dict["end_page"] = page_txt
        else:
            inner_dict["start_page"] = page_loc[0]
            inner_dict["end_page"] = page_loc[-1] 
        chunk_page.append(inner_dict)
    merged_chunks_page = []
    for current_chunk, next_chunk in zip(chunk_page[:-1], chunk_page[1:]):
        # Normalize page info
        current_chunk = normalize_page(current_chunk)
        next_chunk = normalize_page(next_chunk)
        
        # Merge content and calculate page range
        merge_content = current_chunk["chunk"] + next_chunk["chunk"]
        start_page = min(current_chunk["start_page"], next_chunk["start_page"])
        end_page = max(current_chunk["end_page"], next_chunk["end_page"])
        
        # Create merged dictionary and append
        merged_chunks_page.append({
            "chunk": merge_content,
            "start_page": start_page,
            "end_page": end_page
        })


    for index, dic in enumerate(merged_chunks_page):
        if dic["chunk"] == "":
            merged_chunks_page.remove(dic)
            continue

    if merge_tables:
        #應該要修改表格的start_page和end_page，但我想先pass
        pass



    return merged_chunks_page, markdown_to_gptcontent












    #     #若有非OCR的表格，則特別處理
    #     if len(markdown_to_gptcontent) > 0:
    #         for num_table, raw_table in enumerate(table_markdown):
    #             new_page_content = pymupdf_original_text.replace(raw_table, markdown_to_gptcontent[num_table])
    #             if new_page_content == pymupdf_original_text:
    #                 # 如果replace没有成功，新增内容到pymupdf_original_text的最下面(目前版本應該都會替代成功)
    #                 pymupdf_original_text += markdown_to_gptcontent[num_table]
    #             else:
    #                 # 如果replace成功，更新pymupdf_original_text
    #                 pymupdf_original_text = new_page_content
    #         output_text += clean_string(pymupdf_original_text)

    #     elif len(markdown_to_gptcontent) == 0:
    #         output_text += clean_string(pymupdf_original_text)