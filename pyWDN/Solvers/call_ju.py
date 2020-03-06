# import julia
# julia.install()

from julia import Main, SparseArrays

def call_julia(filename,funcname,*args):
    Main.include(filename)
    argjoin=','.join(['args[' + str(i) + ']' for i in range(len(args))])
    return eval('Main.{}({})'.format(funcname,argjoin))
    # return eval('Main.{}(args[0])'.format(funcname))
    # return eval('Main.{}{}'.format(funcname,args.__str__()))

# def call_ju_sp(filename,funcname,x):
#     Main.include(filename)
#     x=SparseArrays.sparse([1],[2],[3])
#     return eval('Main.{}(x)'.format(funcname))


