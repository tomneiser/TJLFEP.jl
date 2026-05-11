module TJLFEPCUDAExt

using CUDA
using TJLFEP
import TJLFEP: TJLF

# Serializes concurrent GPU solves from Threads.@threads workers.
# CUSOLVER handles are task-local but cusolverDnCreate races when 256 threads
# all call it simultaneously on first use. The GPU is already massively parallel
# internally so serializing Julia-level dispatch has no performance cost.
const _GPU_SOLVE_LOCK = ReentrantLock()

function __init__()
    TJLF._CUDA_FUNCTIONAL[] = () -> CUDA.functional()
    TJLF._CUDA_DEVICE_COUNT[] = () -> length(CUDA.devices())
    # Returns Vector{ComplexF64} of eigenvalues (B⁻¹A solved then diagonalised, all on GPU).
    # B is only needed for the LU factorisation; its overwritten form is never used by the caller.
    TJLF._CUDA_SOLVE[] = (A::Matrix{ComplexF64}, B::Matrix{ComplexF64}) -> begin
        lock(_GPU_SOLVE_LOCK) do
            A_gpu = CUDA.CuArray(A)
            B_gpu = CUDA.CuArray(B)
            ipiv = CUDA.CuArray{Int32}(undef, size(B_gpu, 1))
            B_factored, ipiv, _ = CUDA.CUSOLVER.getrf!(B_gpu, ipiv)
            A_gpu = CUDA.CUSOLVER.getrs!('N', B_factored, ipiv, A_gpu)
            W, _, _ = CUDA.CUSOLVER.Xgeev!('N', 'N', A_gpu)
            return Array(W)
        end
    end
end

end
