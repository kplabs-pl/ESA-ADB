import numpy as np
import cv2 as cv
from copy import deepcopy
import pathlib

def convert_index(org_index, max_index, graph_width, margin_left):
    return round(float(org_index/max_index)*graph_width+margin_left)

def draw_csv(ranges, img, h_floor, h_ceiling, color, max_index, graph_width, margin_left):
    for a_range in ranges:
        start_time = convert_index(a_range.get_time()[0], max_index, graph_width, margin_left)
        end_time = convert_index(a_range.get_time()[1], max_index, graph_width, margin_left)
        cv.rectangle(img, (start_time, h_floor), (end_time, h_ceiling), color, thickness=-1)

def draw_csv_range(ranges, img, h_floor, h_ceiling, color, start, end):
    for a_range in ranges:
        if a_range.get_time()[0] <= end or a_range.get_time()[1] >= start:
            cv.rectangle(img, (a_range.get_time()[0]-start+10, h_floor), (a_range.get_time()[1]-start+10, h_ceiling), color, thickness=-1)

def shift_ranges(ranges, first_idx):
    for a_range in ranges:
        a_range.set_time(a_range.get_time()[0] - first_idx, a_range.get_time()[1] - first_idx)

def draw_graphs(anomalies, predictions, how_show: str):
    method_list = [ 'Anomalies', 'Predictions' ]
    anomalies = deepcopy(anomalies)
    predictions = deepcopy(predictions)
    first_idx = min(anomalies[0].get_time()[0]-100, predictions[0].get_time()[0])
    last_idx = max(anomalies[-1].get_time()[1], predictions[-1].get_time()[1])
    marginal_idx = int(float(last_idx-first_idx)/100)
    first_idx -= marginal_idx
    shift_ranges(anomalies, first_idx)
    shift_ranges(predictions, first_idx)
    ranges_list = [ anomalies, predictions ]
    max_index = max(anomalies[-1].get_time()[1], predictions[-1].get_time()[1]) + marginal_idx

    color_list = [(70, 70, 70),  #black
                  (60, 76, 203),  #red
                  (193, 134, 46),  #blue
                  (133, 160, 22),  #green
                  (206, 143, 187),  #purple
                  (94, 73, 52),  # darkblue
                  (63, 208, 244)  #yellow
                  ]

    margin_left   = 10
    margin_right  = 150
    margin_top    = 20
    margin_bottom = 40

    graph_gap = 20
    graph_height = 40
    graph_width = 2000

    n_results = 2

    width = margin_left + graph_width + margin_right
    height = margin_top + margin_bottom + n_results * (graph_gap + graph_height)
    bpp = 3

    img = np.ones((height, width, bpp), np.uint8)*255

    img_h = img.shape[0]
    img_w = img.shape[1]
    img_bpp = img.shape[2]

    thickness = 1
    fontsize = 1
    cv.line(img, (int(margin_left/2), img_h-margin_bottom), (img_w-int(margin_left/2), img_h-margin_bottom), color_list[0], thickness) #x-axis
    pts = np.array([[img_w-int(margin_left/2), img_h-margin_bottom], [img_w-int(margin_left/2)-7, img_h-margin_bottom+5], [img_w-int(margin_left/2)-7, img_h-margin_bottom-5]], np.int32) #arrow_head
    pts = pts.reshape((-1, 1, 2))
    cv.fillPoly(img, [pts], color_list[0])
    cv.putText(img, 'Relative Index', (img_w-180, img_h-15), cv.FONT_HERSHEY_COMPLEX_SMALL, fontsize, color_list[0], 1, cv.LINE_AA) #x-axis label

    for i in range(margin_left, width-margin_right, int(graph_width/10)):
        cv.line(img, (i, img_h-margin_bottom+2), (i, img_h-margin_bottom-2), color_list[0], thickness)
        org_index = str(round((i-10) / graph_width * max_index / 1000))
        cv.putText(img, org_index+'K', (i-len(org_index)*5, img_h-margin_bottom + 25), cv.FONT_HERSHEY_COMPLEX_SMALL, fontsize, color_list[0], 1, cv.LINE_AA)

    thickness = -1
    for idx in range(n_results):
        cv.putText(img, method_list[idx],
                   (width - margin_right + 2, img_h - margin_bottom - graph_gap * (idx+1) - graph_height * idx - 12),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, fontsize, color_list[0], 1, cv.LINE_AA)
        draw_csv(ranges_list[idx], img, h_floor=img_h - margin_bottom - graph_gap * (idx+1) - graph_height * idx,
                 h_ceiling=img_h - margin_bottom - graph_gap * (idx+1) - graph_height * (idx+1),
                 color=color_list[(idx+1)%len(color_list)],
                 max_index=max_index, graph_width=graph_width, margin_left=margin_left)

    if how_show == 'screen' or how_show == 'all':
        cv.imshow("drawing", img)
    if how_show == 'file' or how_show == 'all':
        cv.imwrite("../../brief_result.png", img)
    if how_show != 'screen' and how_show != 'all' and how_show != 'file':
        print('Parameter Error')
    cv.waitKey(0);


def draw_multi_graphs(anomalies, predictions_list, predictions_name_list, how_show: str):
    method_list = [ 'Anomalies' ] + predictions_name_list

    anomalies = deepcopy(anomalies)
    predictions_list = deepcopy(predictions_list)

    first_idx = anomalies[0].get_time()[0]-100
    last_idx = anomalies[-1].get_time()[1]
    for single_prediction in predictions_list:
        first_idx = min(first_idx, single_prediction[0].get_time()[0])
        last_idx = max(last_idx, single_prediction[-1].get_time()[1])

    marginal_idx = int(float(last_idx-first_idx)/100)
    first_idx -= marginal_idx
    shift_ranges(anomalies, first_idx)
    for single_prediction in predictions_list:
        shift_ranges(single_prediction, first_idx)

    ranges_list = [ anomalies ] + predictions_list

    max_index = anomalies[-1].get_time()[1]
    for single_prediction in predictions_list:
        max_index = max(max_index, single_prediction[-1].get_time()[1])
    max_index = max_index + marginal_idx

    color_list = [(0, 0, 0),  #black
                  (60, 76, 203),  #red
                  (193, 134, 46),  #blue
                  (133, 160, 22),  #green
                  (206, 143, 187),  #purple
                  (94, 73, 52),  # darkblue
                  (63, 208, 244)  #yellow
                  ]

    margin_left   = 10
    margin_right  = 180
    margin_top    = 20
    margin_bottom = 40

    graph_gap = 20
    graph_height = 40
    graph_width = 2000

    n_results = len(ranges_list)

    width = margin_left + graph_width + margin_right
    height = margin_top + margin_bottom + n_results * (graph_gap + graph_height)
    bpp = 3

    img = np.ones((height, width, bpp), np.uint8)*255

    img_h = img.shape[0]
    img_w = img.shape[1]
    img_bpp = img.shape[2]

    thickness = 1
    fontsize = 1.4
    cv.line(img, (int(margin_left/2), img_h-margin_bottom), (img_w-int(margin_left/2), img_h-margin_bottom), color_list[0], thickness) #x-axis
    pts = np.array([[img_w-int(margin_left/2), img_h-margin_bottom], [img_w-int(margin_left/2)-7, img_h-margin_bottom+5], [img_w-int(margin_left/2)-7, img_h-margin_bottom-5]], np.int32) #arrow_head
    pts = pts.reshape((-1, 1, 2))
    cv.fillPoly(img, [pts], color_list[0])
    cv.putText(img, 'Relative Index', (img_w-180, img_h-15), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color_list[0], 1, cv.LINE_AA) #x-axis label

    for i in range(margin_left, width-margin_right, int(graph_width/10)):
        cv.line(img, (i, img_h-margin_bottom+2), (i, img_h-margin_bottom-2), color_list[0], thickness)
        org_index = str(round((i-10) / graph_width * max_index / 1000))
        cv.putText(img, org_index+'K', (i-len(org_index)*5, img_h-margin_bottom + 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color_list[0], 1, cv.LINE_AA)

    thickness = -1
    for idx in range(n_results):
        cv.putText(img, method_list[idx],
                   (width - margin_right + 2, img_h - margin_bottom - graph_gap * (idx+1) - graph_height * idx - 12),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, fontsize, color_list[0], 1, cv.LINE_AA)
        draw_csv(ranges_list[idx], img, h_floor=img_h - margin_bottom - graph_gap * (idx+1) - graph_height * idx,
                 h_ceiling=img_h - margin_bottom - graph_gap * (idx+1) - graph_height * (idx+1),
                 color=color_list[(idx+1)%len(color_list)],
                 max_index=max_index, graph_width=graph_width, margin_left=margin_left)

    if how_show == 'screen' or how_show == 'all':
        cv.imshow("drawing", img)
    if how_show == 'file' or how_show == 'all':
        print("The file is saved at " + str(pathlib.Path(__file__).parent.absolute()))
        cv.imwrite("./brief_result.png", img)
    if how_show != 'screen' and how_show != 'all' and how_show != 'file':
        print('Parameter Error')
    cv.waitKey(0);
