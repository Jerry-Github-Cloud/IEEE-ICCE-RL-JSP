import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
from pprint import pprint
from colorhash import ColorHash

from logger import Logger

class Plotter(object):
    def __init__(self, logger):
        self.logger = logger
        
    def _get_tooltip(self, op_info):
        return str(op_info)

    def _get_job_id(self, op_info):
        if 'job_type' in op_info and op_info['job_type'] == 'NOOP':
            return 'NOOP'
        return '%d' %(op_info['job_id'])

    def _get_name(self, op_info):
        return f"O{op_info['job_id']},{op_info['op_id']}"

    def _get_machine(self, op_info):
        return 'Machine %d' %(op_info['machine_id'])

    def _get_color(self, op_info):
        # color = ColorHash(op_info['job_type'])
        # return color.hex
        if op_info['rule_name'] == None:
            return "#C0C0C0"
        color = ColorHash(op_info['rule_name'])
        return color.hex

    def plot_googlechart_timeline(self, html_out_file):
        scale = 10
        history = sorted(self.logger.history, key=lambda op_info: op_info['machine_id'])
        html_text = ''
        html_text += self.logger.google_chart_front_text
        for op_info in history:
            row = [ 
                self._get_machine(op_info), 
                self._get_name(op_info),
                self._get_color(op_info),
                self._get_tooltip(op_info),
                op_info['start_time']*scale, op_info['finish_time']*scale ]
            line = str(row) + ',\n'
            html_text += line
        html_text += self.logger.google_chart_back_text
        with open(html_out_file, 'w') as f:
            f.write(html_text)

    def plot_plotly_timeline(self, html_name, color_by='job_id'):
        ### timeline
        ### x-axis: date
        if isinstance(color_by, str):
            data = self.logger.get_plotly_timeline_input(color_by)
            df = pd.DataFrame(data)
            fig = px.timeline(
                df, x_start='StartDateTime', x_end='FinishDateTime', y='machine_id', color='job_id', 
                hover_name='job_id', hover_data=['job_id', 'op_id', 'process_time', 'Start', 'Finish']
            )
            fig.update_layout(xaxis_type='date')    # ['-', 'linear', 'log', 'date', 'category', 'multicategory']
            fig.write_html(html_name)
        if isinstance(color_by, tuple):
            # colors = [ 'red', 'green', 'blue', 'orange', 'purple' ]
            color_maps = [ 
                px.colors.sequential.Reds[1:],      px.colors.sequential.Greens[1:],    px.colors.sequential.Blues[1:], 
                px.colors.sequential.Greys[1:],   px.colors.sequential.Purples[1:],   px.colors.sequential.Oranges[1:],
                px.colors.sequential.PuRd[1:]]
            color_discrete_map = {}
            for i in range(5):
                for j in range(10):
                    size = len(color_maps[i])
                    color_discrete_map[(i, j)] = color_maps[i][j%size]
            data = self.logger.get_plotly_timeline_input(color_by)
            print(data)
            df = pd.DataFrame(data)
            fig = px.timeline(
                df, x_start='StartDateTime', x_end='FinishDateTime', y='machine_id', color='color', 
                hover_name='job_id', hover_data=['job_id', 'op_id', 'process_time', 'Start', 'Finish'],
                color_discrete_map = color_discrete_map
            )
            fig.update_layout(xaxis_type='date')    # ['-', 'linear', 'log', 'date', 'category', 'multicategory']
            fig.write_html(html_name)

if __name__ == '__main__':
    # result_dir = "../result/instances"
    # timeline_dir = "../timeline/instances"
    # result_dir = "../agent/Rule/result/instances"
    # timeline_dir = "../agent/Rule/timeline/instances"
    # result_dir = "../agent/Rule/result/small_case"
    # timeline_dir = "../agent/Rule/timeline/small_case"
    result_dir = "../agent/SoftDQN/result/20230212_181516"
    timeline_dir = "../agent/SoftDQN/timeline/20230212_181516"
    
    if not os.path.exists(timeline_dir):
        os.makedirs(timeline_dir)
    for i, file_name in enumerate(os.listdir(result_dir)): 
        json_in_file = os.path.join(result_dir, file_name)
        logger = Logger()
        logger.load(json_in_file)
        # logger.find_noop()
        plotter = Plotter(logger)
        fn, _ = os.path.splitext(file_name)
        html_out_file = os.path.join(timeline_dir, fn+'.html')
        print(html_out_file)
        plotter.plot_googlechart_timeline(html_out_file)

