#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_2_kernel(
    const float* Q,
    const float* K,
    const float* V,
    const int N, // Sequence Length
    const int d, // Hidden Dimension per Head
    const int Tc, // Tc = ceil(N / Bc)
    const int Tr, // Tr = ceil(N / Br)
    const int Bc,
    const int Br,
    const float softmax_scale, // = 1 / sqrt(d)
    float* O, // Output Tensor
    const float* startT, // (B * N) start Time
    const float* endT, // (B * N) end Time
    const bool IsTree // If true then use tree causality
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;  // batch, head, tile row index

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];
    
    int i = bz;

    // Load Qi from HBM to SRAM, l and m to registers
    for (int x = 0; x < d; x++) {
        // FIXME: Bank conflict here? Anyway it's not optimal. Adj threads should access adjacent elements.
        Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
    }
    float row_m_prev = -INFINITY;
    float row_l_prev = 0;

    // Causal mask: j <= i
    for (int j = 0; j < Tc; ++j) { // j is the column tile index
        __syncthreads();
        // Load Kj, Vj from HBM to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();

        // S_i^j = softmax_scale * QiKj^T
        // S_i^j[tx][y] = softmax_scale * Sum_{x = 0}^{d-1} Qi[tx][x] * Kj[y][x]
        float row_m = -INFINITY;
        for (int y = 0; y < Bc; y++) {
            int tempI = (bx * N) + i * Br + tx, tempJ = (bx * N) + j * Bc + y; // mask = 1 if I = descendent of J
            bool mask = (!IsTree) || ((startT[tempI] >= startT[tempJ]) && (endT[tempI] <= endT[tempJ]));
            // causal mask
            if (mask){ // FIXME: Thread divergence
                float sum = 0;
                for (int x = 0; x < d; x++)
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x]; // FIXME: Store Kj.T directly. conseq elements.
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum; // FIXME: Again the bank thing. Shouldn't they access conseq elements?

                if (sum > row_m)
                    row_m = sum;
            }
        }

        // m_i^j = max(m_i^j-1, row_max(S_i^j))
        float new_row_m = max(row_m_prev, row_m);

        // P_i^j = exp(S_i^j - m_i^j)
        // P_i^j[tx][y] = exp(S_i^j[tx][y] - m_i^j)
        float row_l = 0;
        for (int y = 0; y < Bc; y++) {
            int tempI = (bx * N) + i * Br + tx, tempJ = (bx * N) + j * Bc + y; // mask = 1 if I = descendent of J
            bool mask = (!IsTree) || ((startT[tempI] >= startT[tempJ]) && (endT[tempI] <= endT[tempJ]));
            // causal mask
            if (mask){ // FIXME: Thread divergence
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - new_row_m);
                row_l += S[(Bc * tx) + y];
            }
        }

        // l_i^j = (exp(m_i^j-1 - m_i^j) * l_i^j-1) + row_sum(P_i^j)
        float row_m_exp = __expf(row_m_prev - new_row_m);
        float new_row_l = (row_m_exp * row_l_prev) + row_l;

        // O_i^j = diag(exp(m_i^j-1 - m_i^j))^-1 * O_i^j-1 + P_i^jVj
        for (int x = 0; x < d; x++) {
            float pv = 0;  // Pij * Vj
            for (int y = 0; y < Bc; y++) {
                int tempI = (bx * N) + i * Br + tx, tempJ = (bx * N) + j * Bc + y; // mask = 1 if I = descendent of J
                bool mask = (!IsTree) || ((startT[tempI] >= startT[tempJ]) && (endT[tempI] <= endT[tempJ]));
                // causal mask
                if (mask){ // FIXME: Thread divergence
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
            }
            O[qkv_offset + (tile_size * i) + (tx * d) + x] = \
                row_m_exp * O[qkv_offset + (tile_size * i) + (tx * d) + x] + pv;
        }

        // Update m and l
        row_m_prev = new_row_m;
        row_l_prev = new_row_l;
    }

    // O_i = diag(l_i^{Tc})^-1 * O_i^{Tc}
    for (int x = 0; x < d; x++)
        O[qkv_offset + (tile_size * i) + (tx * d) + x] /= row_l_prev;
}

std::vector<torch::Tensor> forward_2(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    
    // printf("SMs: %d\n", prop.multiProcessorCount);
    // printf("Max Grid Size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    // printf("Max Block Size: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    // printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    // printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    // printf("Max Warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    // printf("Max Share Memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
    // printf("Max Share Memory per Block: %d KB\n", prop.sharedMemPerBlock / 1024);
    
    // TODO: determine Bc, Br dynamically
    const int Bc = 32;
    const int Br = 32;
    assert(Br == Bc);

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O to HBM
    auto O = torch::zeros_like(Q);
    torch::Device device(torch::kCUDA);

    // Calculate SRAM size needed per block
    int col_tile_size = Bc * d;  // size of Kj, Vj
    int row_tile_size = Br * d;  // size of Qi
    const int sram_size =
        (2 * col_tile_size * sizeof(float))  // SRAM size for Kj, Vj
        + (row_tile_size * sizeof(float))  // SRAM size for Qi
        + (Bc * Br * sizeof(float));  // SRAM size for S
    
    int max_sram_size = prop.sharedMemPerBlock;

    dim3 grid_dim(B, nh, Tr);  // batch_size x num_heads x seq_length
    dim3 block_dim(Br); // FIXME

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    forward_2_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale, O.data_ptr<float>(),
        StartTimes.data_ptr<float>(), EndTimes.data_ptr<float>(), IsTree
    );

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float milliseconds = 0; cudaEventElapsedTime(&milliseconds, start, stop);
    
    return {O, torch::tensor({milliseconds}), torch::tensor({max_sram_size}), torch::tensor({sram_size})};
}
