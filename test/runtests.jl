# TJLFEP test entrypoint.
#
# File-based regression only: we force TJLFEP_FILE_ONLY=1 BEFORE loading TJLFEP so
# the IMAS/FUSE/TurbulentTransport runtime imports are skipped (see TJLFEP.jl).
# Run with:  module load julia/1.11.7 && export JULIA_DEPOT_PATH=$PSCRATCH/.julia
#            julia --project=. -t 8 test/runtests.jl
get(ENV, "TJLFEP_FILE_ONLY", "0") == "1" || (ENV["TJLFEP_FILE_ONLY"] = "1")

using Test

@testset "TJLFEP" begin
    include("runtests_regression_nb6.jl")
end
