# -*- coding: utf-8 -*-
"""
This is the Flesch Readability Score adapted for Russian calculator.
The formula is an adaptation of the Flesch Readability Score for Russian.

This tool can calculate the readability score of a Russian text.
using the Flesch Readability Score adapted for Russian by Oborneva (2006).
https://wp.hse.ru/data/2015/01/12/1106465345/16LNG2014.pdf

Amstad T (1978) Wie verständlich sind unsere Zeitungen? [How readable are our newspapers?]. Doctoral thesis, Universität Zürich, Switzerland

License: GPL-2
"""
from readability_score.textanalyzer import TextAnalyzer


class FleschRussian(TextAnalyzer):
    def __init__(self, text, locale='de_DE'):
        TextAnalyzer.__init__(self,text,locale)
        self.setTextScores()
        self.readingindex = 0
        self.setReadingIndex()

    def setReadingIndex(self):
        self.readingindex = (8.4* self.scores['sentlen_average']) + (0.5 * self.scores['wordlen_average'])-15.59
