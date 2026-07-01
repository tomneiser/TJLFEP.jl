# TJLFEP test entrypoint.
#
# File-based regression only: we force TJLFEP_FILE_ONLY=1 BEFORE loading TJLFEP so
# the IMAS/FUSE/TurbulentTransport runtime imports are skipped (see TJLFEP.jl).
# Run with:  module load julia/1.11.7 && export JULIA_DEPOT_PATH=$PSCRATCH/.julia
#            julia --project=. -t 8 test/runtests.jl
get(ENV, "TJLFEP_FILE_ONLY", "0") == "1" || (ENV["TJLFEP_FILE_ONLY"] = "1")

using Test

@testset "TJLFEP" begin
    # Fast, solve-free unit tests (parsers, structs, file IO) for broad coverage.
    include("unit_helpers.jl")
    include("unit_structs.jl")
    include("unit_io.jl")
    include("unit_extra.jl")
    include("unit_ad_smoke.jl")

    # Small multi-radius scan + finalize (mirror of FUSE's ActorTJLFEP smoke test).
    include("unit_scan_finalize.jl")

    # Ballooning-width scan (PROCESS_IN=3 WIDTH_IN_FLAG=false branch).
    include("unit_widthscan.jl")

    # End-to-end regression against the Fortran TGLF-EP reference (one radius).
    include("runtests_regression_nb6.jl")
    include("runtests_regression_spectrum.jl")
end
