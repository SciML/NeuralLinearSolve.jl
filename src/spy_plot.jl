# Converts a sparse matrix to a 64x64 Float32 grayscale SPY plot for use as input to the CNN solver predictor.
# The SPY plot is a binary image where each pixel indicates the presence of at least one nonzero entry in the corresponding block of the matrix.

using SparseArrays
using Images

const TARGET_SIZE = 64


function matrix_to_spy(A::SparseMatrixCSC)
    # Input: a sparse matrix A
    # Output: 64×64 grayscale spy plot image returned as a Float32 array of shape (64, 64, 1, 1) 
    
    img = _downsample_spy(A, TARGET_SIZE)
    # Shape is (height, width, channels, batch) = (64, 64, 1, 1)
    X = zeros(Float32, TARGET_SIZE, TARGET_SIZE, 1, 1)
    X[:, :, 1, 1] = Float32.(img)
    return X
end

# Helper function
function _downsample_spy(A::SparseMatrixCSC, target_size::Int)
    # Input: a sparse matrix A, target_size for dimensions of SPY plot
    # Output: a target_size x target_size binary matrix
    n, m = size(A)

    if n <= target_size && m <= target_size
        # Upsample via nearest-neighbour
        img = zeros(Bool, target_size, target_size)
        scale_r = target_size / n
        scale_c = target_size / m
        rows = rowvals(A)
        for j in 1:m
            for idx in nzrange(A, j)
                i = rows[idx]
                i_new = min(Int(ceil(i * scale_r)), target_size)
                j_new = min(Int(ceil(j * scale_c)), target_size)
                img[i_new, j_new] = true
            end
        end
        return img
    end

    # Downsample via block aggregation
    block_r = ceil(Int, n / target_size)
    block_c = ceil(Int, m / target_size)
    img = zeros(Bool, target_size, target_size)
    rows = rowvals(A)
    for j in 1:m
        bj = min(div(j - 1, block_c) + 1, target_size)
        for idx in nzrange(A, j)
            i = rows[idx]
            bi = min(div(i - 1, block_r) + 1, target_size)
            img[bi, bj] = true
        end
    end
    return img
end
