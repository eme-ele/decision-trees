import optparse

def parse_and_validate():
    ## parse
    parser = optparse.OptionParser()
    parser.add_option('-i', dest ='max_iterations', type='int',
                      help='(int) max number of iterations. must be > 0', default=10)
    parser.add_option('-r', dest='regularization', type='string',
                      help='(string) l1 or l2 regularization', default='l1')
    parser.add_option('-s', dest='step_size', type='float',
                      help='(float) step size taken by GD. must be > 0 and <= 1', default=0.1)
    parser.add_option('-l', dest='lmbd', type='float',
                      help='(float) lambda argument. must be > 0 and <= 1', default=0.1)
    parser.add_option('-f', dest='feature_set', type='int', default=1,
                      help='(int) 1: unigrams, 2: bigrams, 3: unigrams + bigrams')

    (opts, args) = parser.parse_args()

    ## validate
    if (opts.max_iterations < 0) or\
       (opts.regularization not in ['l1', 'l2']) or\
       (not (opts.step_size > 0 and opts.step_size <= 1)) or\
       (not (opts.lmbd > 0 and opts.lmbd <= 1)) or\
       (opts.feature_set not in [1,2,3]):
           print "\ninvalid argument"
           parser.print_help()
           exit(-1)

    return opts

## Gradient Descent Algorithm
def GD(max_iterations, regularization, step_size, lmbd, feature_set):

    return

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    opts = parse_and_validate()
    print opts.max_iterations, opts.regularization, opts.step_size, opts.lmbd, opts.feature_set

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

if __name__ == '__main__':
    main()
