# coding=utf-8
import re


def reg_match(source, category="unit"):
    assert isinstance(source, str), "the param source must be type of str."
    assert category in {"unit", "result", "range"}, "unsupported category."
    if source == "":
        return None, None
    source = source.replace(" ", "")
    if category == "unit":
        pairs = [(r"[pP][eE][iI1l|][uU]/m[1IlL|]", "PEIU/mL"),
                 (r"10.?3/[1IlL|]?",               "10^3/L"),
                 (r"10.?6/[1IlL|]?",               "10^6/L"),
                 (r"10.?9/[1IlL|]?",               "10^9/L"),
                 (r"10.?12/[1IlL|]?",              "10^12/L"),
                 (r"mm[0oO][1IlL|]/[1IlL|]",       "mmol/L"),
                 (r"[uμ]m[0oO][1IlL|]/[1IlL|]",    "μmol/L"),
                 (r"m[0oO][1IlL|]/[1IlL|]",        "mol/L"),
                 (r"m[I1|][uU]/m[1IlL|]",          "mIU/mL"),
                 (r"[I1|][uU]/m[1IlL|]",           "IU/mL"),
                 (r"mg/d[1IlL|]",                  "mg/dL"),
                 (r"ng/m[1IlL|]",                  "ng/mL"),
                 (r"mg/[1IlL|]",                   "mg/L"),
                 (r"[I1|][uU]/[1IlL|]",            "IU/L"),
                 (r"s/c[o0O]",                     "s/co"),
                 (r"[uU]/[1IlL|]",                 "U/L"),
                 (r"g/[1IlL|]",                    "g/L"),
                 (r"[1IlL|]/[1IlL|]",              "L/L"),
                 (r"/[1IlL|][pP]",                 "/LP"),
                 (r"/[hH][pP]",                    "/HP"),
                 (r"f[1IlL|]",                     "fL"),
                 (r"pg",                           "pg"),
                 ("%",                             "%")]
        for pair in pairs:
            pattern = re.compile(pair[0])
            m = re.search(pattern, source)
            if m is not None and m.group() != "":
                return pair[1], m.span()
    elif category == "result":
        regs = [r"[<>]?-?(\d*[\.,，]\d*|\d*)([\+-]*)([×xX]10.?[1-9]\d*)?",
                r"(\-|\+)?\d+(\.\d+)?"]
        for reg in regs:
            pattern = re.compile(reg)
            m = re.search(pattern, source)
            if m is not None and m.group() != "":
                reg_str = m.group()
                fmt_str = fmt_result(reg_str)
                return fmt_str, m.span()
    elif category == "range":
        regs = [r"-?(\d*[\.,，]\d*|\d*)[-—~]+(\d*[\.,，]\d*|\d*)",
                r"[<>]?-?(\d*[\.,，]\d*|\d*)([×xX]10.?[1-9]\d*)?"]
        for reg in regs:
            pattern = re.compile(reg)
            m = re.search(pattern, source)
            if m is not None and m.group() != "":
                reg_str = m.group()
                return fmt_range(reg_str), m.span()
    return None, None


def fmt_result(source):
    source = fmt_scientific(source)
    origin = source
    try:
        tmp = source
        source = source.replace(",", ".").replace("，", ".").rstrip("+-")
        tail = tmp[len(source):]
        multi = source.find("×")
        if multi >= 0:
            tail = source[multi:] + tail
            source = source[:multi]
        tmp = source
        source = source.lstrip("<>")
        range_sign = tmp[:len(tmp)-len(source)]
        n = float(source)
        if source.find('.') < 0:
            n = int(n)
        return range_sign+str(n)+tail
    except:
        print("can't format result, source=%s" % origin)
    return origin


def fmt_range(source):
    source = fmt_scientific(source)
    source = source.replace("~", "-").replace("—", "-").replace(",", ".").replace("，", ".")
    if source[0] == ">" or source[0] == "<":
        source = source[0] + fmt_result(source[1:])
    else:
        start = source.find("-", 1)  # 可能第一个数为负数
        end = source.rfind("-")
        if start > 0 and end > 0:
            source = fmt_result(source[:start]) + "-" + fmt_result(source[end+1:])
    return source


def fmt_scientific(source):
    """
    格式化科学计数法
    :param source: 原字符串
    :return: 格式化后的字符串
    """
    source = source.replace("x", "×").replace("X", "×")
    multi = source.find("×")
    if multi >= 0 and multi + 3 < len(source):
        if source[multi + 3].isdigit():
            source = source[:multi + 3] + "^" + source[multi + 3:]
        else:
            source = source[:multi + 3] + "^" + source[multi + 4:]
    return source


def item_seg(s):
    value, span = reg_match(s, category="unit")
    if value is not None and value != "":
        print(value, span)
        if span[1] == len(s):
            return value, s[:span[0]], ""
        else:
            return value, s[:span[0]], s[span[1]:]
    return s, ""


if __name__ == "__main__":
    units = ["mmo1/Lmol/l", "umO1/l", "umol/L", "mmo1/L", "mo1/L", "mmol/L", 
             "mg/dL", "10~12/L", "10~9/", "←4.00~10.0(10~9//", "3.50~5.501012/L"]
    match_units = []
    for candi in units:
        res = reg_match(candi, category="unit")
        print(res)
    results = ["123.", "0.312-", "0", "-12.1", ">00012.3", "<1.23 x 103",
               "012,32++", "阴性", "< 1.23x 106", "↑6.44", "←0.3"]
    for result in results:
        res = reg_match(result, category="result")
        print("result correct: [%s] --> [%s]" % (result, res))
    ranges = ["0.00-0.20", "2.00——10.00", "1.003--1.030", "3.5~5", "-12.3~~22.2",
              "<1.00 X 102", "<0.500", ">123", "1.23 X 106"]
    for r in ranges:
        res = reg_match(r, category="range")
        print("range correct: [%s] --> [%s]" % (r, res))
    item_seg("79.6486.0~100.(fl")
