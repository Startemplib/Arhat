import re
import pandas as pd

# 文件路径
markdown_file = "EDFA-Thorlabs-S.md"
excel_file = "Data.xlsx"
readme_file = "Readme.txt"

# 读取 Markdown 文件
with open(markdown_file, "r", encoding="utf-8") as file:
    content = file.read()

if not content.endswith("\n"):
    content += "\n"
content += "# OVER\n"  # 添加结束标记

# 正则表达式：匹配标题
title_pattern = r"^(#+)\s*(.+)$"  # 标题匹配，确保标题后有换行符

# 先按标题分块
lines = content.splitlines()
sections = []
current_hierarchy = []
current_block = {"title_hierarchy": "", "content": []}

for line in lines:
    title_match = re.match(title_pattern, line)
    if title_match:  # 匹配到标题
        # 保存当前块
        if current_block["content"]:
            sections.append(current_block)
            current_block = {"title_hierarchy": "", "content": []}
        # 更新标题层级
        level = len(title_match.group(1))
        title = title_match.group(2).strip()
        while len(current_hierarchy) >= level:
            current_hierarchy.pop()
        current_hierarchy.append(title)
        current_block["title_hierarchy"] = " > ".join(current_hierarchy)
    else:
        # 非标题内容归入当前块，但忽略 <br> 标签
        if "<br>" not in line:
            current_block["content"].append(line)

# 保存最后一个块
if current_block["content"]:
    sections.append(current_block)

# 表格正则表达式
table_pattern = r"(\|.+?\|\n\|[-:| ]+\|\n(?:\|.*?\|\n)+)"  # 匹配 Markdown 表格

# 保存表格到 Excel 和 ReadMe
excel_writer = pd.ExcelWriter(excel_file, engine="xlsxwriter")
table_descriptions = []
marked_content = []  # 保存包含 "mark" 的标题下的内容

print(f"共找到 {len(sections)} 个块。")

for idx, section in enumerate(sections, start=1):
    block_content = "\n".join(section["content"])
    title_hierarchy = section["title_hierarchy"]
    print(f"块 {idx} 的标题层级：{title_hierarchy}")

    # 检查标题中是否包含 "mark"
    if "mark" in title_hierarchy.lower():
        marked_content.append(
            f"标题层级：{title_hierarchy}\n内容：\n{block_content.strip()}\n"
        )

    # 提取表格
    tables = re.findall(table_pattern, block_content)
    if not tables:
        print(f"块 {idx} 中未找到表格。\n")
    else:
        for table_idx, table in enumerate(tables, start=1):
            print(f"块 {idx} 的表格 {table_idx} 内容：\n{table}")

            # 解析表格
            rows = table.strip().split("\n")
            headers = [col.strip() for col in rows[0].split("|") if col.strip()]  # 表头
            data = [
                [col.strip() for col in row.split("|") if col.strip()]
                for row in rows[2:]
            ]  # 数据行

            # 数据广播：对齐表头长度
            broadcasted_data = [
                (
                    row + [""] * (len(headers) - len(row))
                    if len(row) < len(headers)
                    else row
                )
                for row in data
            ]

            # 创建 DataFrame
            df = pd.DataFrame(broadcasted_data, columns=headers)
            print(f"块 {idx} 的表格 {table_idx} 的 DataFrame:")
            print(df)

            # 动态生成表单名
            title_structure = (
                section["title_hierarchy"].replace(">", "-").replace(" ", "")
            )
            sheet_name = f"{title_structure}_T{table_idx}"
            if len(sheet_name) > 31:  # Excel 限制 sheet 名字最长 31 字符
                sheet_name = f"Block_{idx}_T{table_idx}"

            # 写入到 Excel
            df.to_excel(excel_writer, sheet_name=sheet_name, index=False)

            # 描述记录
            description = (
                f"{sheet_name}: 提取自标题层级 "
                f"{title_hierarchy}，包含列：{', '.join(headers)}"
            )
            table_descriptions.append(description)

# 保存 Excel 文件
excel_writer.close()

# 写入 ReadMe 文件
with open(readme_file, "w", encoding="utf-8") as file:
    file.write("Markdown 文件中提取的表格结构描述：\n")
    file.write("\n".join(table_descriptions))
    file.write("\n\n其它信息：\n")
    file.write("\n".join(marked_content))

print(f"数据已导出到 {excel_file}，结构描述已保存到 {readme_file}。")
