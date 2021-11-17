import json
import random
random.seed(7)

sourceFile1 = '../../input/data/ICDAR17_Korean/ufo/train.json'
sourceFile2 = '../../input/data/Camper_Data/ufo/annotation.json'
sourceFile3 = '../../input/data/ReposData/ufo/ufoannotation.json'

targetDir = '../../input/data/'

with open(sourceFile1,'r') as f:
    jsonData_All = json.load(f)
with open(sourceFile2,'r') as f:
    jsonData2 = json.load(f)
with open(sourceFile3,'r') as f:
    jsonData3 = json.load(f)

jsonData_All['images'].update(jsonData2['images'])
jsonData_All['images'].update(jsonData3['images'])

imageNames = list(jsonData_All["images"].keys())
random.shuffle(imageNames)

pivot = int(len(imageNames) * 0.8)
trainNames = imageNames[:pivot]
validNames = imageNames[pivot:]

jsonData_train = {"images":{}}
jsonData_valid = {"images":{}}

for jsonData, names in zip([jsonData_train, jsonData_valid], [trainNames, validNames]):
    for name in names:
        jsonData["images"][name] = jsonData_All["images"][name]

print(len(jsonData_train))
print(len(jsonData_valid))

with open(targetDir+"split_train.json", 'w') as f:
    json.dump(jsonData_train, f, indent="\t")

with open(targetDir+"split_valid.json", 'w') as f:
    json.dump(jsonData_valid, f, indent="\t")