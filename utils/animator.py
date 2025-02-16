#! coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
from .plot import use_svg_display, plot, set_axes
from IPython import display
from IPython.core.interactiveshell import InteractiveShell

matplotlib.use('TkAgg')
matplotlib.rcParams['backend']


class Animator:

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        super(InteractiveShell).__init__()
        # Plot multiple lines incrementlly
        if legend is not None:
            legend = []

        use_svg_display()

        # self.fig, self.axes = plt.subplots(nrows=ncols, ncols=nrows, figsize=figsize)
        self.fig, self.axes = plt.subplots()

        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        # using lambda function for parameter capturing
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

        self.x, self.y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add more datasets into image
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)

        if not hasattr(x, "__len__"):
            x = [x] * n

        if not self.x:
            self.x = [[] for _ in range(n)]

        if not self.y:
            self.y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.x[i].append(a)
                self.y[i].append(b)
        self.axes[0].cla()
        for ax, ay, fmt in zip(self.x, self.y, self.fmts):
            self.axes[0].plot(ax, ay, fmt)
        self.config_axes()

        display.display(self.fig)
        display.clear_output(wait=True)
        # plt.imshow(self.fig)
        # plt.show()
