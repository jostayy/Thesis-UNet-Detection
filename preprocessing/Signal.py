"""
Kod bazowy udostępniony przez promotora.
Wykorzystany do wstępnego przetwarzania sygnałów wibrotermograficznych.
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:55:22 2022

@author: PADuser
"""

class Signal:
    
    def __init__(self,x,y,z,signal):
        self.x = x
        self.y = y
        self.z = z
        self.signal = signal