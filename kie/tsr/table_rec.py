# coding=utf-8
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import math
import time
import cv2
import numpy as np

from collections import deque
from scipy import optimize

from ppocr.utils.logging import get_logger
from kie.utils.nlp import jieba_seg
from kie.utils.u_str import calc_valid_char, str_len
from kie.utils.head_cfg import is_cfg_head_word, is_cfg_key_word, get_category
from kie.utils.xlsx import write_excel_xlsx
from kie.ocr_correct.correct import single_correct, multi_correct

logger = get_logger()


class TableRecognizer(object):
    def __init__(self, args):
        """
        初始化
        :param args: 配置参数
        """
        # TODO: 化验单栏数判断阈值设定
        self.sub_table_thresh = args.sub_table_thresh
        # TODO: 化验单表头提取阈值设定
        self.head_word_seg_thresh = args.head_word_seg_thresh
        # TODO: 化验单表格行分割阈值设定
        self.horizon_line_split_thresh = args.horizon_line_split_thresh
        self.save_path = args.save_path

    def __call__(self, img, img_name, dt_boxes, rec_res, save=True):
        """
        初始化并执行化验单表格内容结构化过程
        :param img: 化验单图像
        :param img_name: 化验单文件名称，不包含后缀，例如 test_sheet (1)
        :param dt_boxes: 检测文本框
        :param rec_res: 识别文本框
        :param save: 是否保存Excel结果
        """
        # init process
        self.img = img
        self.img_name = img_name
        self.dt_boxes = dt_boxes
        self.rec_res = rec_res
        self.text_boxes = {}
        begin = time.time()
        for i in range(len(dt_boxes)):
            text, score = rec_res[i]
            box = dt_boxes[i]
            text_box = TextBox(text, score, box)
            box_key = get_box_key(box)
            self.text_boxes[box_key] = text_box
        end = time.time()
        logger.info("jieba process elapse: {}".format(end-begin))

        # 获取处于同一行的表头文本框列表，可能有多行，[[box_key, ...], ...]
        lines = self.get_same_line_head_boxes()
        line_f_1_list = []  # 表头直线方程列表
        table_cnt_dict = {}  # 记录化验表的栏数（单栏、双栏、三栏...）的dict
        head_line_dict = {}  # 表头直线方程 -> 表头文本框列表
        for line in lines:
            # 根据同一行的表头文本框列表，拟合出表头直线，以 k、b 形式表示，并计算栏数（单栏或双栏），存在无法拟合直线的情况
            k, b, has, sub_table_cnt = self.get_head_line_f_1(line)
            if not has:
                continue
            line_f_1_list.append((k, b))
            line_key = get_line_key(k, b)
            head_line_dict[line_key] = line
            table_cnt_dict[line_key] = sub_table_cnt
        # 将化验单中的所有非表头水平文本框归类，归类依据是属于哪一条表头直线下方
        tables = self.table_classify(line_f_1_list)
        sheets = []  # excel 表格内容，可能包含多个表格
        multi_table_entries = []  # 将excel 表格内容拆分为 {item, result, unit, range} 结构
        raw_items = []  # 为纠错的项目值
        for i, table in enumerate(tables):
            k, b = table["f_1"]
            box_keys = table["box_keys"]
            line_key = get_line_key(k, b)
            sub_table_cnt = table_cnt_dict[line_key]
            logger.info("{} 栏化验单".format(sub_table_cnt))
            horizon_lines = self.split_horizon_lines(box_keys, k, b)  # [box_key, ...]
            self.split_vertical_lines(horizon_lines, head_line_dict[line_key])
            # for idx, line in enumerate(horizon_lines):
            #     print("[LINE %d]............................" % (idx+1))
            #     for key in line:
            #         text_box = self.text_boxes[key]
            #         print("text=%s, corrected=%s, attrs=%s, seg_words=%s, score=%.2f" % (
            #             text_box.text, text_box.corrected, text_box.head_attrs, text_box.seg_words, text_box.score))
            sheet_name = "化验单" + str(i+1) if len(tables) > 1 else "化验单"
            sheet = self.parse_sheet_to_excel(head_line_dict[line_key], horizon_lines, sheet_name, save)
            sheets.append(sheet)
            entries = self.parse_sheet_to_entry(sheet)
            multi_table_entries.append(entries)
            items = self.parse_raw_items(head_line_dict[line_key], horizon_lines)
            raw_items.extend(items)
        return sheets, multi_table_entries, raw_items

    def get_same_line_head_boxes(self):
        """
        提取按行聚类的表头文本框列表，可处理竖直方向多个化验表，水平方向不支持
        :return: [[box_key1, box_key2, ...], ...]
        """
        candi_heads = []
        for box_key, text_box in self.text_boxes.items():
            if len(text_box.head_words) > 0:
                candi_heads.append(box_key)
        same_line_boxes = {}  # box → 与 box 在同一行的 box 列表（不包含 key 对应的 box）
        for outer_key in candi_heads:
            outer_box = self.text_boxes[outer_key].box
            # 确定文本框的直线方程
            k, b = get_box_line_f_1([outer_box])
            same_line_boxes[outer_key] = []
            for inner_key in candi_heads:
                if inner_key == outer_key:
                    continue
                inner_box = self.text_boxes[inner_key].box
                if get_box_line_dire(inner_box, k, b) == 0:
                    same_line_boxes[outer_key].append(inner_key)

        # bfs聚合处于同一行的文本框
        visited = set()

        def box_bfs(key):
            res = []
            queue = deque([key])
            visited.add(key)
            while len(queue) > 0:
                cur_key = queue.popleft()
                res.append(cur_key)
                for tmp_key in same_line_boxes[cur_key]:
                    if tmp_key not in visited:
                        visited.add(tmp_key)
                        queue.append(tmp_key)
            return res

        lines = []
        for tmp_key in same_line_boxes:
            if tmp_key not in visited:
                line = box_bfs(tmp_key)
                lines.append(line)
        # merge line
        lines = sorted(lines, key=lambda t: len(t), reverse=True)
        final_lines, f_1_list = [], []
        for line in lines:
            merged = False
            for idx in range(len(final_lines)):
                k, b = f_1_list[idx][0], f_1_list[idx][1]
                on_line_cnt = 0
                for tmp_key in line:
                    if get_box_line_dire(self.text_boxes[tmp_key].box, k, b) == 0:
                        on_line_cnt += 1
                if on_line_cnt > len(line)//2:
                    final_lines[idx].extend(line)
                    merged = True
                    break
            if merged:
                continue
            if len(line) > 1:
                final_lines.append(line)
                k, b = get_box_line_f_1([self.text_boxes[k].box for k in line])
                f_1_list.append([k, b])
            else:
                # 忽略仅有一个文本框的标题栏
                self.text_boxes[line[0]].is_head = False
                logger.info("head line ignored, text: {}".format(self.text_boxes[line[0]].text))
        # sort the boxes in line
        for i in range(len(final_lines)):
            final_lines[i] = sorted(final_lines[i], key=lambda k: self.text_boxes[k].box[0][0])
        return final_lines

    def get_head_line_f_1(self, box_keys):
        """
        计算表头直线方程
        :param box_keys: box_key列表
        :return: 直线斜率、直线截距、是否可构成直线、化验单子表数量
        """
        # k-斜率，b-截距，has_line-是否可构成直线，sub_table_cnt-化验单子表数量（单栏化验单/双栏化验单）
        k, b, has_line, sub_table_cnt = 0, 0, False, 0
        logger.info("calc head line func begins...")
        boxes, seg_ratio_sum = [], 0
        head_word_nums, word_set = 0, set()
        for key in box_keys:
            box = self.text_boxes[key].box
            boxes.append(box)
            head_word_nums += len(self.text_boxes[key].head_words)
            ch_cnt = 0
            for word in self.text_boxes[key].head_words:
                ch_cnt += len(word)
                if word not in word_set:
                    word_set.add(word)
            seg_ratio_sum += ch_cnt / calc_valid_char(self.text_boxes[key].text)
            logger.info("text: {}, head_words: {}, seg_words: {}".
                        format(self.text_boxes[key].text, self.text_boxes[key].head_words,
                               self.text_boxes[key].seg_words))
        sub_table_ratio = head_word_nums / len(word_set)
        logger.info("head_word_nums: {}, word_set: {}, sub_table_ratio: {:.2f}".
                    format(head_word_nums, word_set, sub_table_ratio))
        if sub_table_ratio - math.floor(sub_table_ratio) >= self.sub_table_thresh:
            sub_table_cnt = int(math.ceil(sub_table_ratio))
        else:
            sub_table_cnt = int(math.floor(sub_table_ratio))
        avg_seg_ratio = seg_ratio_sum / len(boxes)
        if avg_seg_ratio >= self.head_word_seg_thresh:
            k, b = get_box_line_f_1(boxes, self.img)
            has_line = True
            logger.info("calc head line func end, avg_seg_ratio: {:.2f}".format(avg_seg_ratio))
        else:
            for key in box_keys:
                self.text_boxes[key].is_head = False
            logger.info("calc head line func skipped, avg_seg_ratio: {:.2f}".format(avg_seg_ratio))
        return k, b, has_line, sub_table_cnt

    def table_classify(self, line_f_1_list):
        """
        根据表头直线在 y 方向的上下关系，对整个图像的文本框进行归类
        :param line_f_1_list: 直线方程列表，[(k, b), ...]
        :return: [{"f_1": (k, b), "box_keys": [box_key, ...]}, ...]
        """
        tables, box_keys = [], []
        line_box_dire_dict = {}  # {line_key: {box_key: dire, ...}, ...}
        # 直线方程按截距从小到大排序（直线从高到低，x 正方向为 →，y 正方向为 ↓）
        line_f_1_list = sorted(line_f_1_list, key=lambda t: t[1])
        for box_key, text_box in self.text_boxes.items():
            box = text_box.box
            if text_box.is_head or is_vertical(box):
                continue
            box_keys.append(box_key)
            for k, b in line_f_1_list:
                line_key = get_line_key(k, b)
                dire = get_box_line_dire(box, k, b)
                if line_key not in line_box_dire_dict.keys():
                    line_box_dire_dict[line_key] = {box_key: dire}
                else:
                    line_box_dire_dict[line_key][box_key] = dire
        for idx, f_1 in enumerate(line_f_1_list):
            tmp_box_keys = []
            line_key = get_line_key(f_1[0], f_1[1])
            below_lines = []
            if idx + 1 < len(line_f_1_list):
                below_lines = line_f_1_list[idx + 1:]
            for box_key in box_keys:
                # 属于当前化验表的文本框的条件：在当前表头直线下方，并且在剩余直线上方
                if line_box_dire_dict[line_key][box_key] != -1:
                    continue
                flag = True  # 当前文本框是否在剩余直线上方的标志，默认 True
                for l in below_lines:
                    tmp_line_key = get_line_key(l[0], l[1])
                    if line_box_dire_dict[tmp_line_key][box_key] != 1:
                        flag = False
                        break
                if flag:
                    tmp_box_keys.append(box_key)
            table = {"f_1": f_1, "box_keys": tmp_box_keys}
            tables.append(table)
        return tables

    def split_horizon_lines(self, box_keys, k, b):
        """
        分割水平文本行
        :param box_keys: 直线 y = k*x + b 下方所有文本框的键列表
        :param k: 直线斜率
        :param b: 直线截距
        :return: [box_key1, box_key2, ...]
        """
        box_dis = []
        thresh = self.horizon_line_split_thresh
        for key in box_keys:
            box = self.text_boxes[key].box
            p = get_center_point(box)
            dis = get_p_to_l_dis(k, b, p)
            box_dis.append([key, dis])
        box_dis = sorted(box_dis, key=lambda t: t[1])  # 按文本框到直线的距离升序排序
        line_idx = 0  # 行号
        pre_k = k  # 前一条直线斜率
        lines = []  # 每一行的文本框列表
        avg_line_gap, sum_line_gap = 0.0, 0.0  # 行与行之间距离的平均值、合计值
        line_split_fin = False  # 行切割是否完毕
        while True:
            if len(box_dis) == 0:
                break
            line_idx += 1  # 行号
            box_num = 0  # 同一行文本框中的编号
            # height_sum 为同一行文本框的高度累加值，用于计算同一行的文本框的平均高度 avg_height
            height_sum, avg_height = 0.0, 0.0
            pre_dis = box_dis[0][1]  # 前一个文本框到直线的距离
            line_box_keys = []  # 同一行的文本框列表
            line_box_dis_sum = 0  # 同一行的文本框到上一行直线的距离之和
            for idx, val in enumerate(box_dis):
                box_key, dis = val[0], val[1]
                box = self.text_boxes[box_key].box
                delta = dis - pre_dis
                scale = 0
                # 使用当前文本框与前一个文本框到直线距离的差值 占 前面同一行文本框高度平均值的比例 来判断当前文本框是否属于当前行
                if avg_height != 0:
                    scale = delta / avg_height
                # 边界 case，最后一个 box
                if scale < thresh and idx + 1 == len(box_dis):
                    line_box_keys.append(box_key)
                    line_box_dis_sum += dis
                    scale = 1
                if scale > thresh or self.get_top_horizon_overlap(box, line_box_keys, thresh) is not None:
                    # 对判定为同行的文本框从左到右排序
                    line_box_keys = sorted(line_box_keys, key=lambda k: self.text_boxes[k].box[0][0])
                    avg_line_box_dis = line_box_dis_sum / len(line_box_keys)
                    if not line_split_fin:
                        # TODO: 化验单非化验结果行剔除阈值
                        if avg_line_gap != 0.0 and avg_line_box_dis >= 2 * avg_line_gap:
                            line_split_fin = True
                        else:
                            key_word_cnt, tmp_key_words = 0, []
                            for k in line_box_keys:
                                for w in self.text_boxes[k].seg_words:
                                    if is_cfg_key_word(w):
                                        key_word_cnt += 1
                                        tmp_key_words.append(w)
                            if key_word_cnt > 0:
                                logger.info("key_words in current line boxes, key_words: {}".format(tmp_key_words))
                            else:
                                lines.append(line_box_keys)
                                sum_line_gap += avg_line_box_dis
                                avg_line_gap = sum_line_gap / len(lines)
                    # 打印输出
                    # print("[LINE %d]............................" % line_idx)
                    pure_boxes = [self.text_boxes[k].box for k in line_box_keys]
                    # 当一行的文本框不少于2个时，才重新计算该行直线方程；否则沿用上一行的直线斜率
                    _k, _b = get_box_line_f_1(pure_boxes, self.img, default_k=pre_k)
                    pre_k = _k
                    # 计算剩余文本框到该直线的距离
                    box_dis = box_dis[len(line_box_keys):]
                    for pair in box_dis:
                        p = get_center_point(self.text_boxes[pair[0]].box)
                        pair[1] = get_p_to_l_dis(_k, _b, p)
                    box_dis = sorted(box_dis, key=lambda t: t[1])
                    break
                else:
                    # 没有换行，继续累加文本框
                    box_num += 1
                    line_box_keys.append(box_key)
                    line_box_dis_sum += dis
                    height_sum += get_box_height(box)
                    avg_height = height_sum / box_num
                    pre_dis = dis
        return lines

    def split_vertical_lines(self, lines, head_line):
        """
        在水平切分化验栏的基础上，再垂直切分化验栏。会修改入参lines中的meta信息，标记上所属表头类别，如meta["attrs"] = ["NO", "项目"]
        :param lines: 化验表中处于同一水平直线上的文本框，[box_key1, box_key2, ...]
        :param head_line: 表头文本框，[box_key1, box_key2, ...]
        :return: CSV表头列表，例如 ["NO", "项目", "结果", "单位", "参考范围"]
        """
        split_head_boxes = []  # 根据表头文本分词结果进行分割的表头文本框
        csv_head_words = []  # CSV表头
        for box_key in head_line:
            head_words = self.text_boxes[box_key].head_words
            # 记录表头文本框的行列编号，行号只有一个，列号可能有多个
            self.text_boxes[box_key].row = 1
            self.text_boxes[box_key].columns = [len(csv_head_words)+1+x for x in range(0, len(head_words))]
            csv_head_words.extend(head_words)
            split_head_boxes.extend(self.split_head_box(box_key))
        for col, pair in enumerate(split_head_boxes):
            candi_boxes = []
            for row, line in enumerate(lines):
                for k in line:
                    # 记录文本框行号
                    self.text_boxes[k].row = row + 2
                # TODO: 候选文本框筛选阈值
                target_box_key = self.get_top_horizon_overlap(pair["box"], line, thresh=0, mode="AVG")
                if target_box_key is not None:
                    candi_boxes.append(pair["box"])
            if len(candi_boxes) == 0:
                continue
            # TODO: 剔除候选文本框异常值
            merged_box = merge_boxes(candi_boxes)
            for line in lines:
                # TODO: 化验栏文本框分类的重叠阈值
                target_box_key = self.get_top_horizon_overlap(merged_box, line, thresh=0, mode="AVG")
                if target_box_key is not None:
                    # 记录文本框所属列名和列号，可能有多个
                    self.text_boxes[target_box_key].head_attrs.append(pair["head_word"])
                    self.text_boxes[target_box_key].columns.append(col+1)
        self.split_and_correct(lines)
        return csv_head_words

    def split_and_correct(self, lines):
        for line in lines:
            for box_key in line:
                attrs = self.text_boxes[box_key].head_attrs
                if len(attrs) == 0:
                    continue
                source = self.text_boxes[box_key].text
                categories = [get_category(attr) for attr in attrs]
                if len(categories) == 1:
                    # correct single field
                    res, _, _ = single_correct(source, category=categories[0])
                    self.text_boxes[box_key].corrected = [res]
                else:
                    # correct multi field
                    res = multi_correct(source, categories)
                    self.text_boxes[box_key].corrected = res
        return

    def get_top_horizon_overlap(self, box, box_keys, thresh=0.6, mode="MAX"):
        """
        获取一个文本框与文本框数组中有最大水平交集的文本框
        :param box: 文本框（为兼顾该方法会在分割的表头文本框场景使用，不能采用 box_key）
        :param box_keys: 文本框键列表
        :param thresh: 阈值
        :param mode: 取投影交集的模式，支持 "MAX" 和 "AVG" 两种模式
        :return: 文本框键
        """
        ret_box_key, max_ratio = None, 0
        for key in box_keys:
            tmp_box = self.text_boxes[key].box
            ratio = box_horizon_overlap(box, tmp_box, mode=mode)
            if ratio > thresh and ratio > max_ratio:
                max_ratio = ratio
                ret_box_key = key
        return ret_box_key

    def split_head_box(self, box_key):
        """
        分割表头文本框，例如"单位提示"分割为["单位", "提示"]，并以 word 长度占比分割 box
        :param box_key: 文本框键
        :return: [{"head_word": "单位", "box": box}, ...]
        """
        box = self.text_boxes[box_key].box
        head_words = self.text_boxes[box_key].head_words
        if len(head_words) == 1:
            return [{"head_word": head_words[0], "box": box}]
        ratios = []
        total_sum, single_sum = 0, 0
        for w in head_words:
            total_sum += str_len(w)
        for w in head_words:
            single_sum += str_len(w)
            ratios.append(single_sum / total_sum)
        width = get_box_width(box)
        up_k, up_b = fit_line_form_points([box[0], box[1]])
        down_k, down_b = fit_line_form_points([box[2], box[3]])
        points = [(box[0], box[3])]
        for r in ratios:
            x = box[0][0] + width * r
            y = up_k * x + up_b
            p1 = (x, y)
            x = box[3][0] + width * r
            y = down_k * x + down_b
            p2 = (x, y)
            points.append((p1, p2))
        boxes = []
        for i in range(1, len(points)):
            box = [points[i-1][0], points[i][0], points[i][1], points[i-1][1]]
            boxes.append({"head_word": head_words[i-1], "box": box})
        return boxes

    def parse_sheet_to_excel(self, head_box_keys, body_lines, sheet_name="化验单", save=True):
        if len(head_box_keys) == 0 or len(body_lines) == 0:
            logger.info("illegal params to write_excel_sheet!!!")
            return
        data = []
        # table head
        head_data = []
        for key in head_box_keys:
            head_data.extend(self.text_boxes[key].head_words)
        data.append(head_data)
        # table body
        for line in body_lines:
            body_data = [""]*len(head_data)
            for key in line:
                columns = self.text_boxes[key].columns
                corrected = self.text_boxes[key].corrected
                for idx, col in enumerate(columns):
                    body_data[col-1] = corrected[idx]
            data.append(body_data)
        if save:
            write_excel_xlsx(self.save_path, self.img_name, sheet_name, data)
        return data

    def parse_raw_items(self, head_box_keys, body_lines):
        if len(head_box_keys) == 0 or len(body_lines) == 0:
            logger.info("empty content!!!")
            return
        head_data = []
        for key in head_box_keys:
            head_data.extend(self.text_boxes[key].head_words)
        cols, items = set(), []
        for i, word in enumerate(head_data):
            if get_category(word) == "item":
                cols.add(i+1)
        for line in body_lines:
            for key in line:
                columns = self.text_boxes[key].columns
                for idx, col in enumerate(columns):
                    if col in cols:
                        raw_item = self.text_boxes[key].text
                        corrected = self.text_boxes[key].corrected[idx]
                        same = "same" if raw_item == corrected else "corrected"
                        item = raw_item + "\t" + corrected + "\t" + same
                        items.append(item)
                        break
        return items

    def parse_sheet_to_entry(self, sheet):
        """
        将单栏或多栏化验单表格解析为entry形式
        :param sheet: 化验单表格
        :return: {"items": [], "results": [], "units": [], "ranges": []}
        """
        sub_table_cols = []
        heads = sheet[0]
        idx = 0
        while idx < len(heads):
            start_col, end_col = idx, len(heads)-1
            if get_category(heads[idx]) == "item":
                for i in range(start_col+1, len(heads)):
                    if get_category(heads[i]) == "item":
                        end_col = i - 1
                        idx = i - 1
                        break
                sub_table_cols.append([start_col, end_col])
            idx += 1
        entries = []
        for col_pair in sub_table_cols:
            for line in sheet[1:]:
                item, result, unit, ref_range = "", "", "", ""
                for col in range(col_pair[0], col_pair[1]+1):
                    cate = get_category(heads[col])
                    if cate == "item":
                        item = line[col]
                    elif cate == "result":
                        result = line[col]
                    elif cate == "unit":
                        unit = line[col]
                    elif cate == "range":
                        ref_range = line[col]
                if item == "":
                    continue
                entries.append({"item": item, "result": result, "range": ref_range, "unit": unit})
        return entries


class TextBox(object):
    def __init__(self, text, score, box):
        self.text = text
        self.score = score
        self.box = box
        # jieba分词
        seg_words = jieba_seg(text)
        self.seg_words = seg_words  # self.text 的 jieba 分词结果
        self.head_words = []  # self.seg_words 中属于表头的分词列表
        for word in seg_words:
            # 英文单词转换为大写
            upper_word = word.upper()
            if is_cfg_head_word(upper_word):
                self.head_words.append(upper_word)
        self.is_head = len(self.head_words) > 0  # self.text 是否为表头，其值后续可能会被更改
        self.head_attrs = []  # 文本框所属表头类别，如["项目"、"单位"]
        self.corrected = []  # 文本内容针对每一个所属表头类别的纠正结果，如["红细胞数", "10^12/L"]
        self.row = 0  # 文本框所属行，从 1 开始，表头行编号为 1
        self.columns = []  # 文本框所属列，可能有多个值，从 1 开始


# 获取两点之间的直线距离
def get_len(p1, p2):
    x = p1[0] - p2[0]
    y = p1[1] - p2[1]
    return math.sqrt(x*x + y*y)


# 以文本框的左上角顶点的 x, y 坐标组合成文本框的 key
def get_box_key(box):
    return "%d_%d" % (box[0][0], box[0][1])


# 获取文本框左右两边的中点坐标
def get_side_points(box):
    lp = [(box[0][0] + box[3][0]) / 2.0, (box[0][1] + box[3][1]) / 2.0]
    rp = [(box[1][0] + box[2][0]) / 2.0, (box[1][1] + box[2][1]) / 2.0]
    return lp, rp


# 计算文本框的质心坐标（四个顶点坐标的平均值）
def get_center_point(box):
    [x1, y1], [x2, y2] = get_side_points(box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# 获取文本框宽度值
def get_box_width(box):
    return get_len(box[0], box[1])


# 获取文本框高度值
def get_box_height(box):
    return get_len(box[0], box[3])


# 计算文本框的横纵比（高宽比）
def get_aspect_ratio(box):
    ratio = math.fabs(get_box_width(box) / get_box_height(box))
    return ratio


# 计算文本框倾斜角，以角度值表示，位于1、4象限
def get_box_angle(box):
    [x1, y1], [x2, y2] = get_side_points(box)
    theta = math.atan2(y1 - y2, x2 - x1)
    # 将弧度制的角度换算到一、四象限
    if theta > math.pi / 2.0:
        theta -= math.pi
    elif theta < -math.pi / 2.0:
        theta += math.pi
    angle = math.degrees(theta)
    return angle


# 判断文本框是否为竖直
def is_vertical(box):
    ratio = get_aspect_ratio(box)
    angle = get_box_angle(box)
    if ratio < 0.5 or angle > 45 or angle < -45:
        return True
    return False


# 直线方程函数，y = k*x + b
def f_1(x, k, b):
    return k*x + b


def get_line_key(k, b):
    """
    获取直线的 key，用于 dict
    :param k: 斜率
    :param b: 截距
    :return: 字符串 key
    """
    return "%.2f_%.2f" % (k, b)


# 计算点到直线的距离，直线方程 y = k*x + b
def get_p_to_l_dis(k, b, p):
    x, y = p[0], p[1]
    # 将斜截式转为一般式，a*x + b*y + c = 0
    a, b, c = k, -1, b
    return math.fabs(a*x + b*y + c) / math.sqrt(a*a + b*b)


# 计算点投影到直线的点的坐标，直线方程 y = k*x + b
def get_p_to_l_proj(k, b, p):
    x, y = p[0], p[1]
    x_ = (k*y + x - k*b) / (k*k + 1)
    y_ = k*x_ + b
    return x_, y_


def get_box_line_dire(box, k, b):
    """
    判断直线与文本框方位。注意，坐标系建立在图像上，x正方向为 →，y正方向为 ↓
    :param box: 文本框
    :param k: 直线斜率
    :param b: 直线截距
    :return: 1-文本框在直线上方 0-直线穿过文本框 -1-文本框在直线下方
    """
    up, on, down = False, False, False
    for p in box:
        _x, _y = p[0], p[1]
        y = k*_x + b
        if _y > y:
            down = True
        elif _y == y:
            on = True
        else:
            up = True
    if on or (up and down):
        return 0
    elif up and not down:
        return 1
    else:
        return -1


def get_box_line_f_1(boxes, img=None, default_k=0):
    """
    计算多个文本框拟合的直线方程，长（x方向）文本框取左右两边的中点，短文本框取质心
    :param boxes: 文本框列表
    :param img: 图片，传该参数时在图像上绘制直线
    :param default_k: 当拟合直线的点数少于2个时，直线斜率使用默认斜率
    :return: 直线方程参数，k-斜率 b-截距
    """
    x_cors, y_cors = [], []
    draw_x1, draw_x2 = float("inf"), 0
    for box in boxes:
        ratio = get_aspect_ratio(box)
        [x1, y1], [x2, y2] = get_side_points(box)
        if ratio >= 1:
            x_cors.extend([x1, x2])
            y_cors.extend([y1, y2])
        else:
            x_cors.append((x1 + x2) / 2.0)
            y_cors.append((y1 + y2) / 2.0)
        if x1 < draw_x1:
            draw_x1 = x1
        if x2 > draw_x2:
            draw_x2 = x2
    if len(x_cors) >= 2:
        k, b = optimize.curve_fit(f_1, x_cors, y_cors)[0]
    else:
        avg_x, avg_y = np.mean(x_cors), np.mean(y_cors)
        k = default_k
        b = avg_y - k * avg_x
    if img is not None:
        draw_y1, draw_y2 = int(f_1(draw_x1, k, b)), int(f_1(draw_x2, k, b))
        cv2.line(img, (int(draw_x1), int(draw_y1)), (int(draw_x2), int(draw_y2)), (0, 0, 255), 2, cv2.LINE_AA)
    return k, b


def box_horizon_overlap(box1, box2, mode="MAX"):
    """
    计算两个文本框在水平方向投影的交集，与两个文本框投影长度比例的较大值
    :param box1: 文本框1
    :param box2: 文本框2
    :param mode: 返回值取值模式，默认 "MAX"（两个比例中较大值），支持 "AVG"（两个比例的平均值）
    :return: 比例
    """
    lp_1, rp_1 = get_side_points(box1)
    lp_2, rp_2 = get_side_points(box2)
    x_cors = [(lp_1[0]+lp_2[0])/2, (rp_1[0]+rp_2[0])/2]
    y_cors = [(lp_1[1]+lp_2[1])/2, (rp_1[1]+rp_2[1])/2]
    k, b = optimize.curve_fit(f_1, x_cors, y_cors)[0]
    min_p, max_p = [None, None], [None, None]
    for i, box in enumerate([box1, box2]):
        for point in box:
            p = get_p_to_l_proj(k, b, point)
            if min_p[i] is None or p[0] < min_p[i][0]:
                min_p[i] = p
            if max_p[i] is None or p[0] > max_p[i][0]:
                max_p[i] = p
    # min_p 中较大值
    lp = min_p[0] if min_p[0][0] >= min_p[1][0] else min_p[1]
    # max_p 中较小值
    rp = max_p[0] if max_p[0][0] <= max_p[1][0] else max_p[1]
    if lp[0] >= rp[0]:
        return 0
    # 两个文本框在直线上投影后，x方向交集的长度
    common_x_len = rp[0] - lp[0]
    # 文本框1、2在直线上投影后，x方向的长度
    x_len1 = max_p[0][0] - min_p[0][0]
    x_len2 = max_p[1][0] - min_p[1][0]
    ratio1 = common_x_len / x_len1
    ratio2 = common_x_len / x_len2
    if mode == "AVG":
        return (ratio1 + ratio2) / 2
    return ratio1 if ratio1 >= ratio2 else ratio2


def fit_line_form_points(points):
    """
    根据坐标点列表拟合直线方程
    :param points: 坐标点列表，[p1, p2, p3, ...]
    :return: 直线的斜率和截距
    """
    x_cors, y_cors = [], []
    for p in points:
        x_cors.append(p[0])
        y_cors.append(p[1])
    k, b = optimize.curve_fit(f_1, x_cors, y_cors)[0]
    return k, b


def merge_boxes(boxes):
    """
    合并文本框。合并后的文本框的四个顶点坐标为文本框列表对应顶点的平均值
    :param boxes: 文本框列表
    :return: 合并后的文本框
    """
    merged_box = [[0, 0], [0, 0], [0, 0], [0, 0]]
    for box in boxes:
        for i, p in enumerate(box):
            merged_box[i][0] += p[0]
            merged_box[i][1] += p[1]
    for p in merged_box:
        p[0] /= len(boxes)
        p[1] /= len(boxes)
    return merged_box
