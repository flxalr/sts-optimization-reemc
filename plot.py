#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot.py:
script to generate plot and data for sts transitions on REEM-C
"""
__author__ = "Felix Aller"
__copyright__ = "Copyright 2022, Felix Aller"
__license__ = "BSD-2-Clause"
__version__ = "1."
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Production"

import pandas as pd
import numpy as np
import math
import statistics
from scipy.integrate import simps
import csaps
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PROJECT = 'input'
WRITE = True
EXP = ['477_104', '457_104', '436_104', '415_104', '498_104', '498_125', '498_145', '498_166']

EXP_TO_NAME = {
    '415_104': 'P1: 100%',
    '436_104': 'P1: 105%',
    '457_104': 'P1: 110%',
    '477_104': 'P1: 115%',
    '498_104': 'P2: 25%',
    '498_125': 'P2: 30%',
    '498_145': 'P2: 35%',
    '498_166': 'P2: 40%'
}

CONTACT_LOST = {
    '415_104': 9.0953474939219880E-01,
    '436_104': 9.5133214516763454E-01,
    '457_104': 4.1228451607725902E-01,
    '477_104': 3.8537227742140584E-01,
    '498_104': 3.4978569175874774E-01,
    '498_125': 4.1702307906487102E-01,
    '498_145': 5.2303405477110387E-01,
    '498_166': 3.9871766135996461E-01,
}

DURATION = {
    '415_104': 2.21973,
    '436_104': 2.22685,
    '457_104': 1.63638,
    '477_104': 1.58987,
    '498_104': 1.53759,
    '498_125': 1.66291,
    '498_145': 1.78625,
    '498_166': 1.87387
}

AUG_DELTA = {
    '415_104': 9,
    '436_104': 5,
    '457_104': 9,
    '477_104': 8,
    '498_104': 8,
    '498_125': 9,
    '498_145': 6,
    '498_166': 5,
}

CHAIR_START_FRAME = {
    '415_104': 46,
    '436_104': 25,
    '457_104': 28,
    '477_104': 31,
    '498_104': 32,
    '498_125': 60,
    '498_145': 39,
    '498_166': 15
}

MASS = 77.5
LEGLENGTH = 0.859
COLORS = px.colors.qualitative.Dark24


def get_start_pos(data, exp):
    exp = exp
    calib = data.query('time <= 5')
    my_start = 999
    frame = 999
    delta = AUG_DELTA[exp]
    for column in data:
        if column not in ['leg_left_3_joint', 'leg_left_4_joint', 'leg_left_5_joint', 'leg_right_3_joint',
                          'leg_right_4_joint', 'leg_right_5_joint', 'arm_right_1_joint', 'torso_2_joint']:
            continue

        # if std <= 0.05: # set to 0.1 rad = ~ 5deg
        std = 0.03

        my_min = [calib[column] - std][0].min()
        my_min = my_round(my_min, 2, down=True)
        my_max = [calib[column] + std][0].max()
        my_max = my_round(my_max, 2, up=True)

        index_high = (data[column].values >= my_max).argmax()
        index_low = (data[column].values <= my_min).argmax()
        if data.time[index_high] <= my_start and index_high != 0:
            my_start = data.time[index_high - delta]
            frame = index_high - delta
        if data.time[index_low] <= my_start and index_low != 0:
            my_start = data.time[index_low - delta]
            frame = index_low - delta
    return my_start, frame


def my_round(n, decimals=0, up=False, down=False):
    multiplier = 10 ** decimals
    if up:
        return math.ceil(n * multiplier) / multiplier
    elif down:
        return math.floor(n * multiplier) / multiplier


def adjust_augmented_to_time(orig_data, my_global_time, starttime):
    new_data = pd.DataFrame(columns=list(orig_data.columns))

    rounded = np.around(my_global_time, decimals=5)
    a = np.round(starttime, 5)
    index = np.where(rounded == a)
    index = index[0]
    end_time = orig_data.iloc[-1]['time']
    my_global_time = my_global_time[index[0]:]
    global_time_zero = my_global_time - my_global_time[0]
    index = np.where(global_time_zero >= end_time)[0][0]  # - 1
    global_time_zero = global_time_zero[:index]
    my_global_time = my_global_time[:index]
    new_data.loc[:, 'time'] = my_global_time
    mod_data, time = False, False
    for column in orig_data:
        if column == 'time':
            dups = orig_data.time.duplicated()
            index = dups[dups].index.values
            mod_data = orig_data.drop(orig_data.index[index])
            time = mod_data[column].to_numpy()
            continue

        data = mod_data[column].to_numpy()
        spl = csaps.CubicSmoothingSpline(time, data, smooth=1).spline
        if 'Torque' in column:
            if 'Hip' in column or 'Knee' in column or 'Ankle' in column:
                new_data.loc[:, column] = spl(global_time_zero) / 2
            else:
                new_data.loc[:, column] = spl(global_time_zero)
        else:
            new_data.loc[:, column] = spl(global_time_zero)
    return new_data


def adjust_time(exp, data, start):
    duration = DURATION[exp]
    new_data = data.query('time >= @start').copy()
    new_data = new_data.reset_index(drop=True)
    starttime = new_data['time'][0]
    new_data['time'] = new_data['time'] - starttime
    new_data = new_data.query('time <= @duration')
    new_data = new_data.reset_index(drop=True)
    return new_data


def smooth(data):
    df = pd.DataFrame(columns=data.columns.values.tolist())
    time = data['time'].to_numpy()
    df['time'] = time
    for column in data:
        if column == 'time':
            continue

        data_col = data[column].to_numpy()
        spl = csaps.CubicSmoothingSpline(time, data_col).spline
        df[column] = spl(time)

    return df


def fig_update_show(fig, width, height):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=3, r=3, t=10, b=10),
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        font=dict(
            size=12,
            color="black",
        ))
    fig.update_xaxes(showline=False, linewidth=1, linecolor='grey', zerolinecolor='darkgrey', zerolinewidth=0, gridcolor='grey', gridwidth=0)
    fig.update_yaxes(showline=False, linewidth=1, linecolor='grey', zerolinecolor='darkgrey', zerolinewidth=0, gridcolor='grey', gridwidth=0)
    fig.show()


def plot_pos_diff(width, height, my_write=False):
    marker_size = 8
    column_titles = ['Protocol 1 - Lower Chair Height', 'Protocol 2 - Increased Ankle Distance']
    row_titles = ['Hip Joint', 'Knee Joint', 'Ankle Joint', 'Torso Joint', 'Shoulder Joint', 'Elbow Joint']
    fig = make_subplots(rows=6, cols=2, row_titles=row_titles, column_titles=column_titles, horizontal_spacing=0.05, vertical_spacing=0.01, shared_xaxes='columns')

    fig.add_trace(
        go.Scatter(x=[0], y=[-1], name='on robot', line=dict(color='grey', width=2, dash='dash'), mode="lines",
                   showlegend=True), row=1,
        col=1)
    fig.add_trace(
        go.Scatter(x=[0], y=[-1], name='simulation', line=dict(color='grey', width=1), mode="lines", showlegend=True),
        row=1, col=1)
    fig.add_scatter(x=[0.5], y=[-50], name='chair contact lost', line=dict(color='grey', width=0),
                    marker_size=marker_size, marker_color='grey', showlegend=True, row=1, col=1)

    fig.add_scatter(x=[0.5], y=[-50], name='end of motion', line=dict(color='grey', width=0), marker_size=marker_size,
                    marker_color='grey', marker_symbol='line-ns-open', showlegend=True, row=1, col=1)
    i = 0
    for exp in EXP:
        if i <= 3:
            col = 1
        else:
            col = 2
        row = 1
        lost = CONTACT_LOST[exp]
        condition = position_adjusted[exp]["time"] >= lost
        contact_lost = condition.idxmax()
        name1 = EXP_TO_NAME[exp]
        name2 = exp
        joint = 'leg_left_3_joint'
        joint_aug = ' StateHipRotY'
        my_color = COLORS[1 + i]
        y_robot = position_adjusted[exp][joint].to_numpy()
        y_aug = augmented_adjusted[exp][joint_aug].to_numpy()
        fig.add_trace(
            go.Scatter(x=position_adjusted[exp]['time'].to_numpy(), y=y_robot,
                       name=name2, line=dict(color=my_color, width=2, dash='dash'), showlegend=False), row=row,
            col=col)
        fig.add_trace(
            go.Scatter(x=augmented_adjusted[exp]['time'].to_numpy(), y=y_aug,
                       name=name1, line=dict(color=my_color, width=1), showlegend=True), row=row, col=col)
        y = y_robot[contact_lost]
        t = position_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=row,
                        col=col)
        fig.add_scatter(x=[augmented_adjusted[exp].iloc[-1]['time']], y=[y_robot[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)
        fig.add_scatter(x=[position_adjusted[exp].iloc[-1]['time']], y=[y_aug[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)

        row = 2
        joint = 'leg_left_4_joint'
        joint_aug = ' StateKneeRotY'
        y_robot = position_adjusted[exp][joint].to_numpy()
        y_aug = augmented_adjusted[exp][joint_aug].to_numpy()
        fig.add_trace(
            go.Scatter(x=position_adjusted[exp]['time'].to_numpy(), y=y_robot,
                       name=name2, line=dict(color=my_color, width=2, dash='dash'), showlegend=False), row=row,
            col=col)
        fig.add_trace(
            go.Scatter(x=augmented_adjusted[exp]['time'].to_numpy(), y=y_aug,
                       name=name1, line=dict(color=my_color, width=1), showlegend=False), row=row, col=col)
        y = y_robot[contact_lost]
        t = position_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=row,
                        col=col)
        fig.add_scatter(x=[augmented_adjusted[exp].iloc[-1]['time']], y=[y_robot[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)
        fig.add_scatter(x=[position_adjusted[exp].iloc[-1]['time']], y=[y_aug[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)

        row = 3
        joint = 'leg_left_5_joint'
        joint_aug = ' StateAnkleRotY'
        y_robot = position_adjusted[exp][joint].to_numpy()
        y_aug = augmented_adjusted[exp][joint_aug].to_numpy()
        fig.add_trace(
            go.Scatter(x=position_adjusted[exp]['time'].to_numpy(), y=y_robot,
                       name=name2, line=dict(color=my_color, width=2, dash='dash'), showlegend=False), row=row,
            col=col)
        fig.add_trace(
            go.Scatter(x=augmented_adjusted[exp]['time'].to_numpy(), y=y_aug,
                       name=name1, line=dict(color=my_color, width=1), showlegend=False), row=row, col=col)
        y = y_robot[contact_lost]
        t = position_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=row,
                        col=col)
        fig.add_scatter(x=[augmented_adjusted[exp].iloc[-1]['time']], y=[y_robot[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)
        fig.add_scatter(x=[position_adjusted[exp].iloc[-1]['time']], y=[y_aug[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)

        row = 4
        joint = 'torso_2_joint'
        joint_aug = ' StateMiddleTrunkRotY'
        y_robot = position_adjusted[exp][joint].to_numpy()
        y_aug = augmented_adjusted[exp][joint_aug].to_numpy()
        fig.add_trace(
            go.Scatter(x=position_adjusted[exp]['time'].to_numpy(), y=y_robot,
                       name=name2, line=dict(color=my_color, width=2, dash='dash'), showlegend=False), row=row,
            col=col)
        fig.add_trace(
            go.Scatter(x=augmented_adjusted[exp]['time'].to_numpy(), y=y_aug,
                       name=name1, line=dict(color=my_color, width=1), showlegend=False), row=row, col=col)
        y = y_robot[contact_lost]
        t = position_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=row,
                        col=col)
        fig.add_scatter(x=[augmented_adjusted[exp].iloc[-1]['time']], y=[y_robot[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)
        fig.add_scatter(x=[position_adjusted[exp].iloc[-1]['time']], y=[y_aug[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)

        row = 5
        joint = 'arm_left_1_joint'
        joint_aug = ' StateLShoulderRotY'
        y_robot = position_adjusted[exp][joint].to_numpy()
        y_aug = augmented_adjusted[exp][joint_aug].to_numpy()
        fig.add_trace(
            go.Scatter(x=position_adjusted[exp]['time'].to_numpy(), y=y_robot,
                       name=name2, line=dict(color=my_color, width=2, dash='dash'), showlegend=False), row=row,
            col=col)
        fig.add_trace(
            go.Scatter(x=augmented_adjusted[exp]['time'].to_numpy(), y=y_aug,
                       name=name1, line=dict(color=my_color, width=1), showlegend=False), row=row, col=col)
        y = y_robot[contact_lost]
        t = position_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=row,
                        col=col)
        fig.add_scatter(x=[augmented_adjusted[exp].iloc[-1]['time']], y=[y_robot[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)
        fig.add_scatter(x=[position_adjusted[exp].iloc[-1]['time']], y=[y_aug[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)

        row = 6
        joint = 'arm_left_4_joint'
        joint_aug = ' StateLElbowRotY'
        y_robot = position_adjusted[exp][joint].to_numpy()
        y_aug = augmented_adjusted[exp][joint_aug].to_numpy()
        fig.add_trace(
            go.Scatter(x=position_adjusted[exp]['time'].to_numpy(), y=y_robot,
                       name=name2, line=dict(color=my_color, width=2, dash='dash'), showlegend=False), row=row,
            col=col)
        fig.add_trace(
            go.Scatter(x=augmented_adjusted[exp]['time'].to_numpy(), y=y_aug,
                       name=name1, line=dict(color=my_color, width=1), showlegend=False), row=row, col=col)
        y = y_robot[contact_lost]
        t = position_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=row,
                        col=col)
        fig.add_scatter(x=[augmented_adjusted[exp].iloc[-1]['time']], y=[y_robot[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)
        fig.add_scatter(x=[position_adjusted[exp].iloc[-1]['time']], y=[y_aug[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)

        i += 1

    fig.update_xaxes(title_text="Time [s]", row=6, col=1)
    fig.update_xaxes(title_text="Time [s]", row=6, col=2)
    fig.update_yaxes(title_text="Degree [°]", row=1, col=1)
    fig.update_yaxes(title_text="Degree [°]", row=2, col=1)
    fig.update_yaxes(title_text="Degree [°]", row=3, col=1)
    fig.update_yaxes(title_text="Degree [°]", row=4, col=1)
    fig.update_yaxes(title_text="Degree [°]", row=5, col=1)
    fig.update_yaxes(title_text="Degree [°]", row=6, col=1)
    fig_update_show(fig, width, height)
    if my_write is True:
        fig.write_image("images/pos_diff.svg")
    return True


def plot_trq_diff(width, height, my_write=False):
    marker_size = 8
    column_titles = ['Protocol 1 - Lower Chair Height', 'Protocol 2 - Increased Ankle Distance']
    row_titles = ['Hip Joint', 'Knee Joint', 'Ankle Joint', 'Torso Joint']
    fig = make_subplots(rows=4, cols=2, row_titles=row_titles, column_titles=column_titles, horizontal_spacing=0.1)

    fig.add_trace(
        go.Scatter(x=[0], y=[-1], name='on robot', line=dict(color='grey', width=2, dash='dash'), mode="lines",
                   showlegend=True), row=1,
        col=1)
    fig.add_trace(
        go.Scatter(x=[0], y=[-1], name='simulation', line=dict(color='grey', width=1), mode="lines", showlegend=True),
        row=1, col=1)
    fig.add_scatter(x=[0.5], y=[-0.5], name='chair contact lost', line=dict(color='grey', width=0),
                    marker_size=marker_size, marker_color='grey', showlegend=True, row=1, col=1)

    fig.add_scatter(x=[0.5], y=[-0.5], name='end of motion', line=dict(color='grey', width=0), marker_size=marker_size,
                    marker_color='grey', marker_symbol='line-ns-open', showlegend=True, row=1, col=1)
    i = 0
    for exp in EXP:

        if i <= 3:
            col = 1
        else:
            col = 2
        row = 1
        lost = CONTACT_LOST[exp]
        condition = trq_smooth[exp]["time"] >= lost
        contact_lost = condition.idxmax()
        name1 = EXP_TO_NAME[exp]
        name2 = exp
        joint = 'leg_left_3_joint'
        joint_aug = ' StateHipTorque'
        my_color = COLORS[1 + i]
        fig.add_trace(
            go.Scatter(x=trq_smooth[exp]['time'].to_numpy(), y=trq_smooth[exp][joint].to_numpy(),
                       name=name2, line=dict(color=my_color, width=2, dash='dash'), showlegend=False), row=row,
            col=col)
        fig.add_trace(
            go.Scatter(x=augmented_adjusted[exp]['time'].to_numpy(), y=augmented_adjusted[exp][joint_aug].to_numpy(),
                       name=name1, line=dict(color=my_color, width=1), showlegend=True), row=row, col=col)
        y = trq_smooth[exp][joint].to_numpy()[contact_lost]
        t = trq_smooth[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=row,
                        col=col)
        fig.add_scatter(x=[augmented_adjusted[exp].iloc[-1]['time']], y=[augmented_adjusted[exp].iloc[-1][joint_aug]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)
        fig.add_scatter(x=[trq_smooth[exp].iloc[-1]['time']], y=[trq_smooth[exp].iloc[-1][joint]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)

        row = 2
        joint = 'leg_left_4_joint'
        joint_aug = ' StateKneeTorque'
        fig.add_trace(
            go.Scatter(x=trq_smooth[exp]['time'].to_numpy(), y=trq_smooth[exp][joint].to_numpy(),
                       name=name1, line=dict(color=my_color, width=2, dash='dash'), showlegend=False),
            row=row, col=col)
        fig.add_trace(
            go.Scatter(x=augmented_adjusted[exp]['time'].to_numpy(), y=augmented_adjusted[exp][joint_aug].to_numpy(),
                       name=name2, line=dict(color=my_color, width=1), showlegend=False), row=row, col=col)
        y = trq_smooth[exp][joint].to_numpy()[contact_lost]
        t = trq_smooth[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=row,
                        col=col)
        fig.add_scatter(x=[augmented_adjusted[exp].iloc[-1]['time']], y=[augmented_adjusted[exp].iloc[-1][joint_aug]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)
        fig.add_scatter(x=[trq_smooth[exp].iloc[-1]['time']], y=[trq_smooth[exp].iloc[-1][joint]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)

        row = 3
        joint = 'leg_left_5_joint'
        joint_aug = ' StateAnkleTorque'
        fig.add_trace(
            go.Scatter(x=trq_smooth[exp]['time'].to_numpy(), y=trq_smooth[exp][joint].to_numpy(),
                       name=name1, line=dict(color=my_color, width=2, dash='dash'), showlegend=False),
            row=row, col=col)
        fig.add_trace(
            go.Scatter(x=augmented_adjusted[exp]['time'].to_numpy(), y=augmented_adjusted[exp][joint_aug].to_numpy(),
                       name=name2, line=dict(color=my_color, width=1), showlegend=False), row=row, col=col)

        temp = trq_smooth[exp][joint].to_numpy()
        y = temp[contact_lost]
        t = trq_smooth[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=row,
                        col=col)
        fig.add_scatter(x=[augmented_adjusted[exp].iloc[-1]['time']], y=[augmented_adjusted[exp].iloc[-1][joint_aug]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)
        fig.add_scatter(x=[trq_smooth[exp].iloc[-1]['time']], y=[trq_smooth[exp].iloc[-1][joint]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)

        row = 4
        joint = 'torso_2_joint'
        joint_aug = ' StateMiddleTrunkTorque'
        fig.add_trace(
            go.Scatter(x=trq_smooth[exp]['time'].to_numpy(), y=trq_smooth[exp][joint].to_numpy(),
                       name=name1, line=dict(color=my_color, width=2, dash='dash'), showlegend=False),
            row=row, col=col)
        fig.add_trace(
            go.Scatter(x=augmented_adjusted[exp]['time'].to_numpy(), y=augmented_adjusted[exp][joint_aug].to_numpy(),
                       name=name2, line=dict(color=my_color, width=1), showlegend=False), row=row, col=col)
        y = trq_smooth[exp][joint].to_numpy()[contact_lost]
        t = trq_smooth[exp]['time'][contact_lost]

        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=row,
                        col=col)
        fig.add_scatter(x=[augmented_adjusted[exp].iloc[-1]['time']], y=[augmented_adjusted[exp].iloc[-1][joint_aug]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)
        fig.add_scatter(x=[trq_smooth[exp].iloc[-1]['time']], y=[trq_smooth[exp].iloc[-1][joint]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=row, col=col)

        i += 1

    fig_update_show(fig, width, height)
    if my_write is True:
        fig.write_image("images/trq_diff.svg")
    return True


def plot_dist(width, height, my_write=False):
    marker_size = 8
    column_titles = ['Protocol 1 - Lower Chair Height', 'Protocol 2 - Increased Ankle Distance']
    row_titles = ['ZMP-BOS', 'FPE-BOS', 'CAP-BOS', 'COM(x)-Chair', 'Norm. COM',
                  'COM Acceleration']
    fig = make_subplots(rows=5, cols=2, row_titles=row_titles, column_titles=column_titles, horizontal_spacing=0.05, vertical_spacing=0.01, shared_xaxes='columns')

    fig.add_scatter(x=[0.5], y=[0], name='chair contact lost', line=dict(color='grey', width=0),
                    marker_size=marker_size, marker_color='grey', showlegend=True, row=1, col=1)

    fig.add_scatter(x=[0.5], y=[0], name='end of motion', line=dict(color='grey', width=0), marker_size=marker_size,
                    marker_color='grey', marker_symbol='line-ns-open', showlegend=True, row=1, col=1)
    i = 0
    for exp in EXP:
        if i <= 3:
            col = 1
        else:
            col = 2
        lost = CONTACT_LOST[exp]
        condition = zmp_adjusted[exp]["time"] >= lost
        contact_lost = condition.idxmax()
        name1 = EXP_TO_NAME[exp]
        my_color = COLORS[1 + i]
        fig.add_trace(
            go.Scatter(x=zmp_adjusted[exp]['time'].to_numpy(), y=zmp_adjusted[exp]['dist'].to_numpy(),
                       name=name1, line=dict(color=my_color, width=1), showlegend=True), row=1, col=col)
        y = zmp_adjusted[exp]['dist'].to_numpy()[contact_lost]
        t = zmp_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=1,
                        col=col)
        fig.add_scatter(x=[zmp_adjusted[exp].iloc[-1]['time']], y=[zmp_adjusted[exp].iloc[-1]['dist']],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=1, col=col)

        fig.add_trace(
            go.Scatter(x=fpe_adjusted[exp]['time'].to_numpy(), y=fpe_adjusted[exp]['dist'].to_numpy(),
                       name=name1, line=dict(color=my_color, width=1), showlegend=False), row=2, col=col)
        y = fpe_adjusted[exp]['dist'].to_numpy()[contact_lost]
        t = fpe_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=2,
                        col=col)
        fig.add_scatter(x=[fpe_adjusted[exp].iloc[-1]['time']], y=[fpe_adjusted[exp].iloc[-1]['dist']],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=2, col=col)

        fig.add_trace(
            go.Scatter(x=cap_adjusted[exp]['time'].to_numpy(), y=cap_adjusted[exp]['dist'].to_numpy(),
                       name=name1, line=dict(color=my_color, width=1), showlegend=False), row=3, col=col)
        y = cap_adjusted[exp]['dist'].to_numpy()[contact_lost]
        t = cap_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=3,
                        col=col)
        fig.add_scatter(x=[cap_adjusted[exp].iloc[-1]['time']], y=[cap_adjusted[exp].iloc[-1]['dist']],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=3, col=col)

        fig.add_trace(
            go.Scatter(x=com_loc_adjusted[exp]['time'].to_numpy(), y=com_loc_adjusted[exp]['x'].to_numpy(),
                       name=name1, line=dict(color=my_color, width=1), showlegend=False), row=4, col=col)
        y = com_loc_adjusted[exp]['x'].to_numpy()[contact_lost]
        t = com_loc_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=4,
                        col=col)
        fig.add_scatter(x=[com_loc_adjusted[exp].iloc[-1]['time']], y=[com_loc_adjusted[exp].iloc[-1]['x']],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=4, col=col)

        fig.add_trace(
            go.Scatter(x=com_vel_adjusted[exp]['time'].to_numpy(), y=com_vel_adjusted[exp]['vel'].to_numpy(),
                       name=name1, line=dict(color=my_color, width=1), showlegend=False), row=5, col=col)
        y = com_vel_adjusted[exp]['vel'].to_numpy()[contact_lost]
        t = com_vel_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=5,
                        col=col)
        fig.add_scatter(x=[com_vel_adjusted[exp].iloc[-1]['time']], y=[com_vel_adjusted[exp].iloc[-1]['vel']],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=5, col=col)

        i += 1

    fig.update_xaxes(title_text="Time [s]", row=5, col=1)
    fig.update_xaxes(title_text="Time [s]", row=5, col=2)
    fig.update_yaxes(title_text="Distance [cm]", row=1, col=1)
    fig.update_yaxes(title_text="Distance [cm]", row=2, col=1)
    fig.update_yaxes(title_text="Distance [cm]", row=3, col=1)
    fig.update_yaxes(title_text="Distance [cm]", row=4, col=1)
    fig.update_yaxes(title_text="Velocity [m/s]", row=5, col=1)
    fig_update_show(fig, width, height)

    if my_write is True:
        fig.write_image("images/distances.svg")
    return True


def plot_angular(width, height, my_write=False):
    marker_size = 8
    column_titles = ['Protocol 1 - Lower Chair Height', 'Protocol 2 - Increased Ankle Distance']
    row_titles = ['Norm. AM(x)', 'Norm. AM(y)', 'Norm. AM(z)']
    fig = make_subplots(rows=3, cols=2, row_titles=row_titles, column_titles=column_titles, horizontal_spacing=0.08, vertical_spacing=0.02, shared_xaxes='columns')

    fig.add_scatter(x=[0.5], y=[-0.02], name='chair contact lost', line=dict(color='grey', width=0),
                    marker_size=marker_size, marker_color='grey', showlegend=True, row=1, col=1)

    fig.add_scatter(x=[0.5], y=[-0.02], name='end of motion', line=dict(color='grey', width=0), marker_size=marker_size,
                    marker_color='grey', marker_symbol='line-ns-open', showlegend=True, row=1, col=1)
    i = 0
    for exp in EXP:
        if i <= 3:
            col = 1
        else:
            col = 2
        lost = CONTACT_LOST[exp]
        condition = com_ang_adjusted[exp]["time"] >= lost
        contact_lost = condition.idxmax()
        name1 = EXP_TO_NAME[exp]
        my_color = COLORS[1 + i]
        y_val = com_ang_adjusted[exp]['x'].to_numpy()
        fig.add_trace(
            go.Scatter(x=com_ang_adjusted[exp]['time'].to_numpy(), y=y_val,
                       name=name1, line=dict(color=my_color, width=1), showlegend=True), row=1, col=col)
        y = y_val[contact_lost]
        t = com_ang_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=1,
                        col=col)
        fig.add_scatter(x=[com_ang_adjusted[exp].iloc[-1]['time']], y=[y_val[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=1, col=col)

        y_val = com_ang_adjusted[exp]['y'].to_numpy()
        fig.add_trace(
            go.Scatter(x=com_ang_adjusted[exp]['time'].to_numpy(), y=y_val,
                       name=name1, line=dict(color=my_color, width=1), showlegend=False), row=2, col=col)
        y = y_val[contact_lost]
        t = com_ang_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=2,
                        col=col)
        fig.add_scatter(x=[com_ang_adjusted[exp].iloc[-1]['time']], y=[y_val[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=2, col=col)

        y_val = com_ang_adjusted[exp]['z'].to_numpy()
        fig.add_trace(
            go.Scatter(x=com_ang_adjusted[exp]['time'].to_numpy(), y=y_val,
                       name=name1, line=dict(color=my_color, width=1), showlegend=False), row=3, col=col)
        y = y_val[contact_lost]
        t = com_ang_adjusted[exp]['time'][contact_lost]
        fig.add_scatter(x=[t], y=[y], marker_size=marker_size, marker_color=my_color, showlegend=False, row=3,
                        col=col)
        fig.add_scatter(x=[com_ang_adjusted[exp].iloc[-1]['time']], y=[y_val[-1]],
                        marker_size=marker_size, marker_color=my_color, marker_symbol='line-ns-open', showlegend=False,
                        row=3, col=col)

        i += 1

    fig.update_xaxes(title_text="Time [s]", row=3, col=1)
    fig.update_xaxes(title_text="Time [s]", row=3, col=2)
    fig.update_yaxes(title_text="[1/s]", row=1, col=1)
    fig.update_yaxes(title_text="[1/s]", row=2, col=1)
    fig.update_yaxes(title_text="[1/s]", row=3, col=1)
    fig_update_show(fig, width, height)
    if my_write is True:
        fig.write_image("images/angular.svg")
    return True


def calc_pos_diff(my_write=False):
    joints = ['leg_left_3_joint', 'leg_left_4_joint', 'leg_left_5_joint', 'torso_2_joint', 'arm_left_1_joint',
              'arm_left_4_joint']
    df = pd.DataFrame(columns=joints, index=EXP)
    for exp in EXP:
        augmented_exp = augmented_adjusted[exp]
        robot = position_adjusted[exp]
        duration = DURATION[exp]
        delta = {}
        column_name_trq = False
        for joint in joints:
            if joint in ['leg_left_3_joint', 'leg_right_3_joint']:
                column_name_trq = ' StateHipRotY'
            elif joint in ['leg_left_4_joint', 'leg_right_4_joint']:
                column_name_trq = ' StateKneeRotY'
            elif joint in ['leg_left_5_joint', 'leg_right_5_joint']:
                column_name_trq = ' StateAnkleRotY'
            elif joint in ['arm_left_1_joint', 'arm_right_1_joint']:
                column_name_trq = ' StateLShoulderRotY'
            elif joint in ['arm_left_4_joint', 'arm_right_4_joint']:
                column_name_trq = ' StateLElbowRotY'
            elif joint == 'torso_2_joint':
                column_name_trq = ' StateMiddleTrunkRotY'

            joint_data_robot = robot[joint].to_numpy()
            joint_data_sim = augmented_exp[column_name_trq].to_numpy()

            delta_single = []
            for j in range(len(joint_data_sim)):
                delta_single.append(joint_data_robot[j] - joint_data_sim[j])
            delta[joint] = abs(np.array(delta_single))
            # l2_value = round(np.sum(np.power((joint_data_robot / duration - joint_data_sim / duration), 2)), 2)
            l2_value = round(np.sqrt(simps(pow(delta[joint], 2))) / duration, 2)
            avg_std = [round(elem, 2) for elem in [statistics.mean(delta[joint]), statistics.stdev(delta[joint])]]
            df.at[exp, joint] = avg_std, l2_value
    if my_write:
        df.to_csv("output/pos_diff.csv")
    return True


def calc_metric_pi(my_data, my_column, name, my_write=False):
    pi = ['avg', 'percentage', 'min', 'max', 'integral']

    round_dec = 2
    if name in ['ang_x_metrics', 'ang_y_metrics', 'ang_z_metrics']:
        round_dec = 4
    result_complete, result_before_c, result_after_c = [], [], []

    df = pd.DataFrame(columns=EXP, index=pi)
    for e in EXP:
        data = my_data[e]

        for column in data:
            if column != my_column:
                continue
            data_column = data[column].to_numpy()

            for p in pi:

                if p == 'min':
                    result_complete = round(min(data_column), round_dec)

                elif p == 'max':
                    result_complete = round(max(data_column), round_dec)

                elif p == 'avg':
                    if name in ['ang_x_metrics', 'ang_y_metrics', 'ang_z_metrics']:
                        result_complete = [round(elem, round_dec) for elem in [statistics.mean(abs(data_column)), statistics.stdev(abs(data_column))]]
                    else:
                        result_complete = [round(elem, round_dec) for elem in [statistics.mean(data_column), statistics.stdev(data_column)]]

                elif p == 'integral':
                    result_complete = round(simps(abs(data_column) / DURATION[e]), round_dec)

                elif p == 'percentage':
                    pos_complete = len([x for x in data_column if x >= 0]) / len(data_column) * 100
                    result_complete = [round(elem, round_dec) for elem in [pos_complete, 100 - pos_complete]]
                df.at[p, e] = result_complete
    if my_write:
        df.to_csv("output/" + name + ".csv")
    return True


def calc_impact(name, my_write=False):
    experiments = ['477_104', '457_104', '436_104', '415_104', '498_104', '498_125', '498_145', '498_166']
    pi = ['min', 'max']
    value = 0
    data = None
    df = pd.DataFrame(columns=experiments, index=pi)
    for e in experiments:
        if name == 'robot':
            data = impact_adjusted[e]
        elif name == 'chair':
            data = chair[e]
        for column in data:
            if column not in ['impact', 'z']:
                continue
            my_data = data[column].to_numpy()
            for p in pi:
                if p == 'min':
                    value = min(my_data)
                elif p == 'max':
                    value = max(my_data)
                df.at[p, e] = round(value, 2)
    if my_write:
        df.to_csv("output/" + name + ".csv")

    return True


if __name__ == '__main__':

    start = {}
    position, position_adjusted = {}, {}
    augmented, augmented_adjusted = {}, {}
    fpe, fpe_adjusted = {}, {}
    cap, cap_adjusted = {}, {}
    zmp, zmp_adjusted = {}, {}
    com_ang, com_ang_adjusted = {}, {}
    com_loc, com_loc_adjusted = {}, {}
    com_vel, com_vel_adjusted = {}, {}
    com_acc, com_acc_adjusted = {}, {}
    trq, trq_adjusted, trq_smooth = {}, {}, {}
    impact, impact_adjusted = {}, {}

    chair = {}
    write = WRITE
    joint_names = ['time', 'base_link_tx', 'base_link_ty', 'base_link_tz', 'base_link_rz', 'base_link_ry',
                   'base_link_rx', 'leg_left_1_joint', 'leg_left_2_joint', 'leg_left_3_joint', 'leg_left_4_joint',
                   'leg_left_5_joint', 'leg_left_6_joint', 'leg_right_1_joint', 'leg_right_2_joint',
                   'leg_right_3_joint', 'leg_right_4_joint', 'leg_right_5_joint', 'leg_right_6_joint', 'torso_1_joint',
                   'torso_2_joint', 'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
                   'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint', 'arm_right_1_joint', 'arm_right_2_joint',
                   'arm_right_3_joint', 'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
                   'arm_right_7_joint', 'head_1_joint', 'head_2_joint']

    for item in EXP:
        file = PROJECT + '/' + item + '/'
        file_nt = PROJECT + '/' + item + '/'
        file_chair = PROJECT + '/' + item + '/'

        position[item] = pd.read_csv(file + 'pos.csv', sep=';')
        global_time = position[item]['time'].to_numpy()
        start[item] = get_start_pos(position[item], item)
        position_adjusted[item] = adjust_time(item, position[item], start[item][0])
        header = position_adjusted[item].columns.values.tolist()
        for h in header:
            if h == 'time':
                continue
            else:
                position_adjusted[item][h] = np.rad2deg(position_adjusted[item][h].to_numpy())

        augmented[item] = pd.read_csv(file_nt + 'augmented.txt', sep=',')
        augmented_adjusted[item] = adjust_augmented_to_time(augmented[item], global_time, start[item][0])
        augmented_adjusted[item] = adjust_time(item, augmented_adjusted[item], start[item][0])
        header = augmented_adjusted[item].columns.values.tolist()
        for h in header:
            if h == 'time':
                continue
            else:
                augmented_adjusted[item][h] = np.rad2deg(augmented_adjusted[item][h].to_numpy())

        fpe[item] = pd.read_csv(file + 'foot_placement_estimator_dist.csv', names=['time', 'dist'], sep=',')
        fpe_adjusted[item] = adjust_time(item, fpe[item], start[item][0])
        header = fpe_adjusted[item].columns.values.tolist()
        for h in header:
            if h == 'time':
                continue
            else:
                fpe_adjusted[item][h] = fpe_adjusted[item][h].to_numpy()*100

        cap[item] = pd.read_csv(file + 'capture_point_dist.csv', names=['time', 'dist'], sep=',')
        cap_adjusted[item] = adjust_time(item, cap[item], start[item][0])
        header = cap_adjusted[item].columns.values.tolist()
        for h in header:
            if h == 'time':
                continue
            else:
                cap_adjusted[item][h] = cap_adjusted[item][h].to_numpy()*100

        zmp[item] = pd.read_csv(file + 'zero_moment_point_dist.csv', names=['time', 'dist'], sep=',')
        zmp_adjusted[item] = adjust_time(item, zmp[item], start[item][0])
        header = zmp_adjusted[item].columns.values.tolist()
        for h in header:
            if h == 'time':
                continue
            else:
                zmp_adjusted[item][h] = zmp_adjusted[item][h].to_numpy()*100

        com_ang[item] = pd.read_csv(file + 'center_of_mass_ang.csv', names=['time', 'x', 'y', 'z'], sep=',')
        com_ang_adjusted[item] = adjust_time(item, com_ang[item], start[item][0])
        header = com_ang_adjusted[item].columns.values.tolist()
        for h in header:
            if h == 'time':
                continue
            else:
                com_ang_adjusted[item][h] = com_ang_adjusted[item][h].to_numpy() / (MASS * pow(LEGLENGTH, 2))

        com_loc[item] = pd.read_csv(file + 'center_of_mass_loc.csv', names=['time', 'x', 'y', 'z'], sep=',')
        com_loc_adjusted[item] = adjust_time(item, com_loc[item], start[item][0])
        header = com_loc_adjusted[item].columns.values.tolist()
        for h in header:
            if h == 'time':
                continue
            else:
                offset = com_loc_adjusted[item][h][0] * 100
                com_loc_adjusted[item][h] = com_loc_adjusted[item][h].to_numpy()*100 - offset

        com_vel[item] = pd.read_csv(file + 'center_of_mass_vel.csv', names=['time', 'vel'], sep=',')
        com_vel_adjusted[item] = adjust_time(item, com_vel[item], start[item][0])

        com_acc[item] = pd.read_csv(file + 'center_of_mass_acc.csv', names=['time', 'acc'], sep=',')
        com_acc_adjusted[item] = adjust_time(item, com_acc[item], start[item][0])

        trq[item] = pd.read_csv(file + 'trq.csv', names=joint_names, sep=';', skiprows=[0])
        trq_adjusted[item] = adjust_time(item, trq[item], start[item][0])
        trq_smooth[item] = smooth(trq_adjusted[item])

        impact[item] = pd.read_csv(file + 'vertical_impact_impact.csv', names=['time', 'impact'], sep=',')
        impact_adjusted[item] = adjust_time(item, impact[item], start[item][0])

        chair[item] = pd.read_csv(file_chair + 'chair_z_pre.csv', names=['sec', 'nsec', 'z'], sep=';', skiprows=[0])

    if plot_pos_diff(800, 960, my_write=write):
        print('pos diff plots created')

    if plot_dist(800, 800, my_write=write):
        print('zmp plots created')

    if plot_angular(800, 480, my_write=write):
        print('ang mom plots created')

    if calc_impact('chair', my_write=write):
        print('robot impact metrics calculated')

    if calc_impact('robot', my_write=write):
        print('robot impact metrics calculated')

    if calc_pos_diff(my_write=write):
        print('pos diff metrics calculated')

    metric_name = 'fpe_metrics'
    if calc_metric_pi(fpe_adjusted, 'dist', metric_name, my_write=write):
        print(metric_name + ' calculated')

    metric_name = 'zmp_metrics'
    if calc_metric_pi(zmp_adjusted, 'dist', metric_name, my_write=write):
        print(metric_name + ' calculated')

    metric_name = 'cap_metrics'
    if calc_metric_pi(cap_adjusted, 'dist', metric_name, my_write=write):
        print(metric_name + ' calculated')

    metric_name = 'ang_x_metrics'
    if calc_metric_pi(com_ang_adjusted, 'x', metric_name, my_write=write):
        print(metric_name + ' calculated')

    metric_name = 'ang_y_metrics'
    if calc_metric_pi(com_ang_adjusted, 'y', metric_name, my_write=write):
        print(metric_name + ' calculated')

    metric_name = 'ang_z_metrics'
    if calc_metric_pi(com_ang_adjusted, 'z', metric_name, my_write=write):
        print(metric_name + ' calculated')

    metric_name = 'com_loc_x_metrics'
    if calc_metric_pi(com_loc_adjusted, 'x', metric_name, my_write=write):
        print(metric_name + ' calculated')

    metric_name = 'com_vel_metrics'
    if calc_metric_pi(com_vel_adjusted, 'vel', metric_name, my_write=write):
        print(metric_name + ' calculated')

    print('finished')
