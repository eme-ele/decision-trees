import sys

# template.py
# -------
# YOUR NAME HERE

##predict a single example
def predict_one(weights, input_snippet):
	pass
	return sign

##Perceptron
#-----------
def perceptron(maxIterations, featureSet):
    pass
    return weights 


##Winnow
#-------
def winnow(maxIterations, featureSet):
    pass
    return weights 

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
	parseArgs([ 'template.py', '-a', 1, '-i', 10, '-f', 1 ]) = {'-t':1, '-i':10, '-f':1 }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

def validateInput(args):
    args_map = parseArgs(args)

    algorithm = 1 # 1: perceptron, 2: winnow
    maxIterations = 10 # the maximum number of iterations. should be a positive integer
	featureSet = 1 # 1: original attribute, 2: pairs of attributes, 3: both

    if '-a' in args_map:
      algorithm = int(args_map['-a'])
    if '-i' in args_map:
      maxIterations = int(args_map['-i'])
    if '-f' in args_map:
      featureSet = int(args_map['-f'])

    assert algorithm in [1, 2]
    assert maxIterations > 0
    assert featureSet in [1, 2, 3]
	
    return [algorithm, maxIterations, featureSet]

def main():
    arguments = validateInput(sys.argv)
    algorithm, maxIterations, featureSet = arguments
    print algorithm, maxIterations, featureSet

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

if __name__ == '__main__':
    main()
