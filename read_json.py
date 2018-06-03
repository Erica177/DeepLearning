# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:08:17 2018

@author: 帅老板
"""

import json as js

def resolveJson(path):
    file = open(path, "rb")
    fileJson = js.load(file)
    field = fileJson["field"]
    futures = fileJson["futures"]
    type = fileJson["type"]
    name = fileJson["name"]
    time = fileJson["time"]

    return (field, futures, type, name, time)

def output():
    result = resolveJson(path)
    print(result)
    for x in result:
        for y in x:
            print(y)


path = r"F:\class\DeepLearning\work\json\run_.-tag-accuracy_2.json"
output()