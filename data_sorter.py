#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:27:54 2021

@author: adernild
"""
#%% Libraries
import os
import shutil
import splitfolders

# Set path to this file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%% Opdeling af hunde og katte
# Path til data
dir = 'data/'

# Looper gennem filer
for file in os.listdir(dir):
    dir_name = file[0:3] # subsetter de første tre karaktere "cat" eller "dog"
    
    # Opretter path til cat og dogs
    dir_path = dir + dir_name
    
    # Tjekker om path eksisterer (hvis ikke skabes det)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Hvis/når det eksisterer tilføjes filen
    if os.path.exists(dir_path):
        file_path = dir + file
        
        shutil.move(file_path, dir_path)

os.remove('data/ico') # Fjerner mappen ico, der indeholder en .ico fil

#%% Splitter mapper i test, train og val
splitfolders.ratio(dir, output=dname, seed=1338, ratio=(.8, .1, .1))


