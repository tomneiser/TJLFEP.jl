
using Pkg
Pkg.activate("../..")
Pkg.resolve()
Pkg.instantiate()
using TJLFEP

TJLFEP.make_crit_grad_plots("neither"; scale="identity", code="fortran")
