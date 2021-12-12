# coding=utf-8
import os
import openpyxl
import pandas as pd

__all__ = ["write_excel_xlsx"]


def write_excel_xlsx(path, xlsx_name, sheet_name, data):
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = os.path.join(path, xlsx_name + ".xlsx")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            sheet.cell(row=i+1, column=j+1, value=str(data[i][j]))
    workbook.save(file_name)
    print("write test sheet to %s successful..." % file_name)


def read_excel_xlsx():
    data = pd.read_excel("./doc/dict/medical_lab_report.xlsx")
    items = set()
    line_cnt = 0
    for line in data.values:
        if line[0][:4] == "test":
            continue
        line_cnt += 1
        if line[0] in items:
            print("dup item: %s" % line[0])
        else:
            items.add(line[0])
    print("len(items)=%d, lines=%d" % (len(items), line_cnt))
    print(items)


if __name__ == "__main__":
    read_excel_xlsx()
