#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:22:45 2017

@author: sahand
"""

import DTree
import sys
import Node
import math
import Node

#find item in a list
def find(item, list):
    for i in list:
        if item(i): 
            return True
        else:
            return False

#find most common value for an attribute
def majority(attributes, data, target):
    #find target attribute
    valFreq = {}
    #find target in data
    index = attributes.index(target)
    #calculate frequency of values in target attr
    for tuple in data:
        if (valFreq.has_key(tuple[index])):
            valFreq[tuple[index]] += 1 
        else:
            valFreq[tuple[index]] = 1
    max = 0
    major = ""
    for key in valFreq.keys():
        if valFreq[key]>max:
            max = valFreq[key]
            major = key
    return major

#Calculates the entropy of the given data set for the target attr
def entropy(attributes, data, targetAttr):

    valFreq = {}
    dataEntropy = 0.0
    
    #find index of the target attribute
    i = 0
    for entry in attributes:
        if (targetAttr == entry):
            break
        ++i
    
    # Calculate the frequency of each of the values in the target attr
    for entry in data:
        if (valFreq.has_key(entry[i])):
            valFreq[entry[i]] += 1.0
        else:
            valFreq[entry[i]]  = 1.0

    # Calculate the entropy of the data for the target attr
    for freq in valFreq.values():
        dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
    return dataEntropy

# Calculates the information gain (reduction in entropy) that would
# result by splitting the data on the chosen attribute (attr).  
def gain(attributes, data, attr, targetAttr):
    valFreq = {}
    subsetEntropy = 0.0
    
    #find index of the attribute
    i = attributes.index(attr)

    # Calculate the frequency of each of the values in the target attribute
    for entry in data:
        if (valFreq.has_key(entry[i])):
            valFreq[entry[i]] += 1.0
        else:
            valFreq[entry[i]] = 1.0
    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in valFreq.keys():
        valProb        = valFreq[val] / sum(valFreq.values())
        dataSubset     = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)
    return (entropy(attributes, data, targetAttr) - subsetEntropy)

#choose best attibute 
def chooseAttr(data, attributes, target):
    best = attributes[0]
    maxGain = 0;
    for attr in attributes:
        newGain = gain(attributes, data, attr, target) 
        if newGain>maxGain:
            maxGain = newGain
            best = attr
    return best

#get values in the column of the given attribute 
def getValues(data, attributes, attr):
    index = attributes.index(attr)
    values = []
    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])
    return values

def getExamples(data, attributes, best, val):
    examples = [[]]
    index = attributes.index(best)
    for entry in data:
        #find entries with the give value
        if (entry[index] == val):
            newEntry = []
            #add value if it is not in best column
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            examples.append(newEntry)
    examples.remove([])
    return examples
    
def DFTreePrint(tree, height, file):
	className = tree.keys()[0]
	valueTree = tree[className]
	values = valueTree.keys()
	for value in values:
		result = valueTree[value]
		if isinstance(result, dict):
			modelString = (height * '| ') + className + ' = ' + value + ' : ' + '\n'
			file.write(modelString)
			DFTreePrint(result, height + 1, file)
		else: 
			modelString = (height * '| ') + className + ' = ' + value + ' : ' + result + '\n'
			file.write((height * '| ') + className + ' = ' + value + ' : ' + result + '\n')

def makeTree(data, attributes, target, recursion):
    recursion += 1
    #Returns a new decision tree based on the examples given.
    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    default = majority(attributes, data, target)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = chooseAttr(data, attributes, target)
        tree = {best:{}}
    
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in getValues(data, attributes, best):
            # Create a subtree for the current value under the "best" field
            examples = getExamples(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = makeTree(examples, newAttr, target, recursion)
    
            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree
#            print tree
			
    return tree


class Node:
    value = ""
    children = []
    
    def __init__(self, val, dictionary):
        self.setValue(val)
        self.genChildren(dictionary)
    
    def __str__(self):
        return str(self.value)
    
    def setValue(self, val):
        self.value = val
        
    def genChildren(self, dictionary):
        if(isinstance(dictionary, dict)):
            self.children = dictionary.keys()


def run(trainData,testData):
	print("training data file: " + str(trainData) )
	file = open(str(trainData))
	"""
	IMPORTANT: Change this variable too change target attribute 
	"""
	target = "Class"
	#extract training data
	data = [[]]
	for line in file:
		line = line.strip("\r\n")
		data.append(line.split(','))
	data.remove([])
	attributes = data[0]
	data.remove(attributes)
	#ID3 DTree
	tree = DTree.makeTree(data, attributes, target, 0)
	file.close()
	print ("generated decision tree")
	f = open(str(sys.argv[3])+ ".model", 'w')
	DTree.DFTreePrint(tree, 0, f)
	f.close()
	data = [[]]
	print ("test data file: " + str(testData))
	f = open(str(testData))
	#extract test data
	for line in f:
		line = line.strip("\r\n")
		data.append(line.split(','))
	data.remove([])
	count = 0
	hits = 0
	total = 0
	for entry in data:
		count += 1
		if str(entry[len(entry)-1]) != target:
			total += 1
			tempDict = tree.copy()
			result = ""
			while(isinstance(tempDict, dict)):
				root = Node.Node(tempDict.keys()[0], tempDict[tempDict.keys()[0]])
				oldDict = tempDict
				tempDict = tempDict[tempDict.keys()[0]]
				index = attributes.index(root.value)
				value = entry[index]
				if str(index) == str(value):
					if str(value) == str(entry[len(entry)-1]):
						hits+=1
						break
			
				child = Node.Node(value, tempDict[value])
				result = tempDict[value]
				tempDict = tempDict[value]
				break
					
	print ("hits: " + str(hits))
	print ("total: " + str(total))
	accuracy = float(hits)/total * 100
	print (str(accuracy) + '%' + ' accuracy on test set')
	
main()