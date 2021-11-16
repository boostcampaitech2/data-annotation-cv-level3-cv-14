import json
import random
random.seed(7)

sourceFile = '../input/data/ICDAR17_Korean/ufo/train.json'
targetDir = '../input/data/ICDAR17_Korean/ufo/'

with open(sourceFile,'r') as f:
    jsonData_All = json.load(f)

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