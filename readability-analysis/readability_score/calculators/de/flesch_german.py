# -*- coding: utf-8 -*-
"""
This is the Flesch Readability Score adapted for German calculator.
The formula is an adaptation of the Flesch Readability Score for German.

This tool can calculate the readability score of a German text.
using the Flesch Readability Score adapted for German by Amstad (1978).
https://link.springer.com/article/10.1007/s13187-018-1358-0

Amstad T (1978) Wie verständlich sind unsere Zeitungen? [How readable are our newspapers?]. Doctoral thesis, Universität Zürich, Switzerland

License: GPL-2
"""
from readability_score.textanalyzer import TextAnalyzer


class FleschGerman(TextAnalyzer):
    def __init__(self, text, locale='de_DE'):
        TextAnalyzer.__init__(self,text,locale)
        self.setTextScores()
        self.readingindex = 0
        self.setReadingIndex()

    def setReadingIndex(self):

        self.readingindex = 180 - self.scores['sentlen_average'] - (58.5 * self.scores['wordlen_average'])
