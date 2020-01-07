def printtree(tree, indent=''):
    # Is this a leaf node?
    if tree.probs is not None:
        print(indent+str(tree.probs))
    else:
        # Print the criteria
        print(indent + str(tree.col)+':'+str(tree.value)+'? ')
        # Print the branches
        print(indent+'T->')
        printtree(tree.tc, indent+'  ')
        print(indent+'F->')
        printtree(tree.fc, indent+'  ')
